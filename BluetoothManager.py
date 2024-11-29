import asyncio
from bleak import BleakClient, BleakScanner
import bluetooth
import time
import json
import numpy as np
import threading
from datetime import datetime, timedelta
import tensorflow as tf
from scipy.signal import spectrogram
from scipy.ndimage import zoom

async def scan_ble_devices():
    devices_list = []
    try:
        print("Scanning devices...")
        devices = await BleakScanner.discover()
        for device in devices:
            device_name = device.name
            if device_name is not None:
                devices_list.append([device_name, device.address])
                print(f"Device: {device_name}, Address: {device.address}")
    except Exception as e:
        print(f"Error: {e}. Please ensure Bluetooth is available and enabled on your device.")
    return devices_list

class SeizureDetector:
    def __init__(self, model_path='seizure_detection_model.keras'):
        print("Loading seizure detection model...")
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = 0.96
        self.prediction_window = []
        self.window_size = 10
        self.consecutive_detections = 0
        self.min_consecutive_detections = 3
        self.last_alert_time = None
        self.alert_cooldown = timedelta(seconds=30)
        print("Model loaded successfully")
        
    def preprocess_data(self, eeg_data):
        """Preprocess EEG data chunk exactly as done during training"""
        try:
            # Ensure the data is the right shape and type
            eeg_data = np.array(eeg_data, dtype=np.float32)
            if len(eeg_data) < 5000:
            # Pad if less than 5000 samples
                eeg_data = np.pad(eeg_data, (0, 5000 - len(eeg_data)), mode='constant')
            elif len(eeg_data) > 5000:
                # Truncate if more than 5000 samples
                eeg_data = eeg_data[:5000]

            
            # Calculate spectrogram with exactly the same parameters as training
            f, t, Sxx = spectrogram(
                eeg_data,
                fs=5000,  # Match sampling rate used in training
                nperseg=256,  # Match training window size
                noverlap=128,  # Match training overlap
                return_onesided=False  # Match training parameter
            )
            
            # Take log and normalize exactly as in training
            Sxx = np.log1p(Sxx)
            Sxx = Sxx / np.max(Sxx) if np.max(Sxx) != 0 else Sxx
            
            # Ensure we have correct dimensions (256, 22)
            if Sxx.shape[0] != 22 or Sxx.shape[1] != 256:
                # Resize to match expected dimensions
                Sxx = zoom(Sxx, (22/Sxx.shape[0], 256/Sxx.shape[1]), order=1)
            
            # Reshape for model input (match training shape)
            return Sxx.T.reshape(1, 256, 22, 1)
            
        except Exception as e:
            print(f"Error in preprocess_data: {str(e)}")
            raise
        
    def predict(self, eeg_data):
        """Make seizure prediction on EEG data chunk"""
        try:
            processed_data = self.preprocess_data(eeg_data)
            prediction = self.model.predict(processed_data, verbose=0)
            prob = float(prediction[0][1])
            
            # Update moving average window
            self.prediction_window.append(prob)
            if len(self.prediction_window) > self.window_size:
                self.prediction_window.pop(0)
            
            # Get average probability over window
            avg_prob = np.mean(self.prediction_window)
            
            # Update consecutive detections
            if avg_prob > self.threshold:
                self.consecutive_detections += 1
            else:
                self.consecutive_detections = 0
                
            # Check if this is a true detection and not in cooldown
            current_time = datetime.now()
            in_cooldown = (
                self.last_alert_time is not None and 
                current_time - self.last_alert_time < self.alert_cooldown
            )
            
            true_detection = (
                self.consecutive_detections >= self.min_consecutive_detections and 
                not in_cooldown
            )
            
            if true_detection:
                self.last_alert_time = current_time
                
            return {
                "probability": avg_prob,
                "detection": true_detection,
                "consecutive_detections": self.consecutive_detections,
                "in_cooldown": in_cooldown
            }
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            raise
        
class ESP32BLEManager:
    def __init__(self, esp32_mac):
        self.esp32_mac = esp32_mac
        self.client = None
        self.connected = False
        self.lock = threading.Lock()
        self.notification_characteristic = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        self.write_characteristic = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
        self.loop = None
        self.connection_thread = None
        self.should_retry = True

    async def connect_with_retry(self):
        """Continuously attempt to connect to ESP32 with retry logic"""
        while self.should_retry:
            if not self.connected:
                try:
                    print(f"Attempting to connect to ESP32: {self.esp32_mac}")
                    device = await BleakScanner.find_device_by_address(self.esp32_mac, timeout=20.0)
                    
                    if device:
                        self.client = BleakClient(device)
                        await self.client.connect()
                        self.connected = True
                        print("Successfully connected to ESP32")
                    else:
                        print("ESP32 device not found, retrying in 5 seconds...")
                        await asyncio.sleep(5)
                        
                except Exception as e:
                    print(f"Connection error: {str(e)}, retrying in 5 seconds...")
                    await asyncio.sleep(5)
            else:
                await asyncio.sleep(1)
    
    def start_connection_thread(self, loop):
        """Start a background thread for ESP32 connection attempts"""
        async def run_connection():
            await self.connect_with_retry()

        asyncio.run_coroutine_threadsafe(run_connection(), loop)

    async def send_alert(self, message="SEIZURE_DETECTED\n"):
        """Send alert to ESP32"""
        if not self.connected or not self.client:
            print("Not connected to ESP32 - Alert not sent")
            return False

        try:
            await self.client.write_gatt_char(
                self.write_characteristic, 
                message.encode(),
                response=True
            )
            print(f"Alert sent to ESP32: {message.strip()}")
            return True
        except Exception as e:
            print(f"Failed to send alert: {str(e)}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from ESP32"""
        self.should_retry = False
        if self.client and self.connected:
            await self.client.disconnect()
            self.connected = False
            print("Disconnected from ESP32")

class AsyncioThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.loop = None
        self.started = threading.Event()
        self.daemon = True

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.started.set()
        self.loop.run_forever()

def handle_client(client_sock, client_info, board, esp32_manager, seizure_detector):
    print(f"Handling connection from {client_info}")
    
    # Ganglion scaling factor from BrainFlow docs
    # Convert from raw ADC to microvolts
    SCALE_FACTOR_EEG = (1.2 * 1000000) / (8388607.0 * 1.5 * 51.0) * 10
    
    # Initialize buffer for collecting samples
    buffer = []
    last_sent_time = time.time()
    
    try:
        while True:
            # Get new data
            data = board.get_current_board_data(10)
            
            if len(buffer) > 10000:  # 2x window size
                print("\nWarning: Buffer overflow, resetting buffer")
                buffer = buffer[-5000:]  # Keep only the most recent window
            
            if data is not None and data.size > 0:
                # Extract channel 3 (index 2) data and add to buffer
                new_samples = data[2, :].tolist()
                buffer.extend([val * SCALE_FACTOR_EEG for val in new_samples])
                
                # Process when we have enough data (5000 samples)
                if len(buffer) >= 5000:
                    try:
                        # Get 5000 samples for processing
                        channel3_data = buffer[:5000]
                        
                        # Slide buffer with 50% overlap (2500 samples)
                        buffer = buffer[2500:]
                        
                        # Optional: Basic artifact rejection
                        channel3_data = [
                            0 if abs(val) > 1000 else val 
                            for val in channel3_data
                        ]
                        
                        # Use the ML model for seizure detection
                        detection_result = seizure_detector.predict(channel3_data)
                        
                        # Create JSON payload
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        json_data = json.dumps({
                            "data": channel3_data[:10],  # First 10 samples for visualization
                            "seizure_detected": detection_result["detection"],
                            "seizure_probability": detection_result["probability"],
                            "consecutive_detections": detection_result["consecutive_detections"],
                            "in_cooldown": detection_result["in_cooldown"],
                            "timestamp": timestamp
                        })
                        
                        # Send data to client (limit rate to once per second)
                        current_time = time.time()
                        if current_time - last_sent_time >= 1.0:
                            client_sock.send(json_data.encode())
                            last_sent_time = current_time
                        
                        # Send alert to ESP32 if seizure detected
                        if detection_result["detection"] and esp32_manager.connected:
                            future = asyncio.run_coroutine_threadsafe(
                                esp32_manager.send_alert(),
                                esp32_manager.loop
                            )
                            try:
                                future.result(timeout=5)
                            except Exception as e:
                                print(f"Failed to send alert: {str(e)}")
                        
                        # Print status update
                        cooldown_str = " (in cooldown)" if detection_result["in_cooldown"] else ""
                        print(f"\rBuffer size: {len(buffer)} | "
                              f"Latest value: {channel3_data[0]:.2f}µV | "
                              f"Probability: {detection_result['probability']:.3f} | "
                              f"Consecutive: {detection_result['consecutive_detections']}"
                              f"{cooldown_str}", end="")
                        
                    except Exception as e:
                        print(f"\nError processing data: {str(e)}")
                        continue
            
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
            
    except Exception as e:
        print(f"\nError in handle_client: {str(e)}")
    finally:
        client_sock.close()
        print(f"\nConnection closed with {client_info}")
        
        
def apply_bandpass_filter(data, sampling_rate=200):
    """
    Apply bandpass filter to EEG data
    Sampling rate for Ganglion is 200Hz by default
    """
    try:
        from brainflow.data_filter import DataFilter, FilterTypes
        
        # Apply bandpass filter (0.5-50Hz is typical for EEG)
        DataFilter.perform_bandpass(
            data,
            sampling_rate,
            2.0,  # center frequency
            4.0,  # band width
            4,    # order
            FilterTypes.BUTTERWORTH.value,
            0     # ripple
        )
        return data
    except Exception as e:
        print(f"Error applying filter: {e}")
        return data
    
def start_bluetooth_server(board):
    # Initialize ESP32 manager
    esp32_mac = "08:B6:1F:B9:36:76"  # Your ESP32's MAC address
    
    # Create and start asyncio thread
    asyncio_thread = AsyncioThread()
    asyncio_thread.start()
    asyncio_thread.started.wait()
    
    # Initialize ESP32 manager and seizure detector
    esp32_manager = ESP32BLEManager(esp32_mac)
    esp32_manager.loop = asyncio_thread.loop
    seizure_detector = SeizureDetector()
    
    # Start ESP32 connection attempts in background
    esp32_manager.start_connection_thread(asyncio_thread.loop)

    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    port = bluetooth.PORT_ANY
    server_sock.bind(("", port))
    server_sock.listen(5)

    uuid = "00001101-0000-1000-8000-00805F9B34FB"
    bluetooth.advertise_service(server_sock, "BrainWaveServer",
                              service_id=uuid,
                              service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                              profiles=[bluetooth.SERIAL_PORT_PROFILE])

    local_address = bluetooth.read_local_bdaddr()
    print(f"Server started on Bluetooth address: {local_address}")
    print(f"Waiting for connections on RFCOMM channel {server_sock.getsockname()[1]}")

    try:
        while True:
            client_sock, client_info = server_sock.accept()
            print(f"Accepted connection from {client_info}")
            client_thread = threading.Thread(
                target=handle_client, 
                args=(client_sock, client_info, board, esp32_manager, seizure_detector)
            )
            client_thread.start()
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        # Clean up ESP32 connection
        future = asyncio.run_coroutine_threadsafe(
            esp32_manager.disconnect(),
            asyncio_thread.loop
        )
        try:
            future.result(timeout=5)
        except Exception as e:
            print(f"Error during ESP32 disconnect: {str(e)}")
            
        # Stop asyncio loop
        asyncio_thread.loop.call_soon_threadsafe(asyncio_thread.loop.stop)
        server_sock.close()
        print("Server stopped")
