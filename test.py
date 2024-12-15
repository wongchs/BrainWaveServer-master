import tensorflow as tf
import numpy as np
import scipy
import json
import time
import threading
import bluetooth
import asyncio
from datetime import datetime, timedelta
from scipy.signal import spectrogram
from bleak import BleakClient, BleakScanner
import os

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
            

class SeizureDetector:
    def __init__(self, model_path='seizure_detection_model.keras'):
        print("Loading seizure detection model...")
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = 0.8
        self.prediction_window = []
        self.window_size = 10
        print("Model loaded successfully")
        
    def preprocess_data(self, eeg_data):
        """Preprocess EEG data chunk exactly as done during training"""
        try:
            # [Previous preprocess_data implementation remains the same...]
            # Ensure the data is the right shape and type
            eeg_data = np.array(eeg_data, dtype=np.float32)
            if len(eeg_data.shape) > 1:
                eeg_data = eeg_data.flatten()
            
            # Basic normalization
            eeg_data = (eeg_data - np.mean(eeg_data)) / (np.std(eeg_data) + 1e-8)
            
            # Calculate spectrogram with fixed parameters
            nperseg = 256
            noverlap = nperseg // 2
            
            f, t, Sxx = spectrogram(
                eeg_data,
                fs=5000,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=256,
                window='hann',
                mode='magnitude'
            )
            
            # [Rest of the existing preprocess_data implementation...]
            if Sxx.shape[0] != 22:
                target_freqs = np.linspace(f[0], f[-1], 22)
                new_Sxx = np.zeros((22, Sxx.shape[1]))
                
                for i in range(22):
                    if i == 21:
                        freq_mask = f >= target_freqs[i]
                    else:
                        freq_mask = (f >= target_freqs[i]) & (f < target_freqs[i + 1])
                    
                    if np.any(freq_mask):
                        new_Sxx[i] = np.mean(Sxx[freq_mask], axis=0)
                    else:
                        nearest_idx = np.abs(f - target_freqs[i]).argmin()
                        new_Sxx[i] = Sxx[nearest_idx]
                
                Sxx = new_Sxx
            
            if Sxx.shape[1] != 256:
                from scipy.ndimage import zoom
                zoom_factor = (1, 256/Sxx.shape[1])
                Sxx = zoom(Sxx, zoom_factor, order=1)
            
            Sxx = np.log1p(np.abs(Sxx))
            Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx) + 1e-8)
            
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
            
            return np.mean(self.prediction_window)
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            raise

class EEGDataReader:
    def __init__(self, patient_num, segment_type='preictal', segment_num=1):
        # self.base_path = f'./kaggle/input/seizure-prediction/Patient_{patient_num}/Patient_{patient_num}/'
        # self.file_path = os.path.join(
        #     self.base_path, 
        #     f'Patient_{patient_num}_{segment_type}_segment_{str(segment_num).zfill(4)}.mat'
        # )
        
        self.base_path = f'./kaggle/input/seizure-prediction/Dog_{patient_num}/Dog_{patient_num}/'
        self.file_path = os.path.join(
            self.base_path, 
            f'Dog_{patient_num}_{segment_type}_segment_{str(segment_num).zfill(4)}.mat'
        )

        self.current_position = 0
        self.load_data()
        
    def load_data(self):
        """Load EEG data from .mat file"""
        try:
            print(f"Loading EEG data from {self.file_path}")
            mat_data = scipy.io.loadmat(self.file_path)
            key = list(mat_data.keys())[-1]  # Get the last key containing data
            self.data = mat_data[key][0][0][0]
            print(f"EEG data loaded successfully. Shape: {self.data.shape}")
            
            # Data quality checks
            if np.any(np.isnan(self.data)):
                print("Warning: NaN values found in EEG data")
            if np.any(np.isinf(self.data)):
                print("Warning: Infinite values found in EEG data")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
    def get_chunk(self, chunk_size=5000):
        """Get next chunk of EEG data (1 second at 5000 Hz)"""
        try:
            if self.current_position + chunk_size > len(self.data[0]):
                self.current_position = 0
                
            chunk = self.data[0][self.current_position:self.current_position + chunk_size]
            self.current_position += chunk_size
            return chunk
        except Exception as e:
            print(f"Error getting chunk: {str(e)}")
            raise

class SeizureAlertServer:
    def __init__(self, detector, data_reader, esp32_manager,
                 data_interval=1.0,          
                 sampling_window=5000,       
                 max_preview_samples=10):    
        self.detector = detector
        self.data_reader = data_reader
        self.esp32_manager = esp32_manager
        self.alert_history = []
        self.last_alert_time = None
        self.alert_cooldown = timedelta(seconds=30)
        self.data_interval = data_interval
        self.sampling_window = sampling_window
        self.max_preview_samples = max_preview_samples
        
    def handle_client(self, client_sock, client_info):
        """Handle individual client connection with real-time seizure detection"""
        print(f"Handling connection from {client_info}")
        consecutive_detections = 0
        min_consecutive_detections = 3
        
        try:
            while True:
                loop_start_time = time.time()
                
                # Get EEG data chunk
                data_chunk = self.data_reader.get_chunk(self.sampling_window)
                
                # Get seizure probability from model
                seizure_prob = self.detector.predict(data_chunk)
                
                # Update consecutive detections
                if seizure_prob > self.detector.threshold:
                    consecutive_detections += 1
                else:
                    consecutive_detections = 0
                
                current_time = datetime.now()
                in_cooldown = (
                    self.last_alert_time is not None and 
                    current_time - self.last_alert_time < self.alert_cooldown
                )
                
                true_detection = (
                    consecutive_detections >= min_consecutive_detections and 
                    not in_cooldown
                )
                
                # Prepare data for sending
                alert_data = {
                    "data": data_chunk[:self.max_preview_samples].tolist(),
                    "seizure_probability": float(seizure_prob),
                    "seizure_detected": true_detection,
                    "consecutive_detections": consecutive_detections,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "in_cooldown": in_cooldown,
                    "sampling_rate": f"{1/self.data_interval:.2f} Hz"
                }
                
                # Send alert to ESP32 if seizure detected
                if true_detection:
                    self.last_alert_time = current_time
                    print(f"\n⚠️ SEIZURE ALERT! Probability: {seizure_prob:.3f}")
                    print(f"Timestamp: {alert_data['timestamp']}")
                    
                    # Send alert to ESP32
                    if self.esp32_manager.connected:
                        future = asyncio.run_coroutine_threadsafe(
                            self.esp32_manager.send_alert(),
                            self.esp32_manager.loop
                        )
                        try:
                            future.result(timeout=5)
                        except Exception as e:
                            print(f"Failed to send ESP32 alert: {str(e)}")
                
                self.alert_history.append(alert_data)
                
                # Send data to client
                json_data = json.dumps(alert_data)
                client_sock.send(json_data.encode())
                
                # Print status with cooldown information
                cooldown_remaining = ""
                if in_cooldown:
                    remaining = self.alert_cooldown - (current_time - self.last_alert_time)
                    cooldown_remaining = f" (Cooldown: {remaining.seconds}s remaining)"
                
                print(f"\rCurrent seizure probability: {seizure_prob:.3f} "
                      f"(Consecutive detections: {consecutive_detections})"
                      f"{cooldown_remaining}", end="")
                
                # Maintain consistent interval
                processing_time = time.time() - loop_start_time
                wait_time = max(0, self.data_interval - processing_time)
                time.sleep(wait_time)
                
        except Exception as e:
            print(f"\nError handling client {client_info}: {str(e)}")
        finally:
            client_sock.close()
            print(f"\nConnection closed with {client_info}")
    
    def start(self):
        """Start the Bluetooth server"""
        server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        port = bluetooth.PORT_ANY
        server_sock.bind(("", port))
        server_sock.listen(5)

        uuid = "00001101-0000-1000-8000-00805F9B34FB"
        bluetooth.advertise_service(
            server_sock, 
            "SeizureAlertServer",
            service_id=uuid,
            service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
            profiles=[bluetooth.SERIAL_PORT_PROFILE]
        )

        local_address = bluetooth.read_local_bdaddr()
        print(f"Server started on Bluetooth address: {local_address}")
        print(f"Waiting for connections on RFCOMM channel {server_sock.getsockname()[1]}")

        try:
            while True:
                client_sock, client_info = server_sock.accept()
                print(f"Accepted connection from {client_info}")
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_sock, client_info)
                )
                client_thread.start()
        except KeyboardInterrupt:
            print("\nServer stopped by user")
        finally:
            # Clean up ESP32 connection
            future = asyncio.run_coroutine_threadsafe(
                self.esp32_manager.disconnect(),
                self.esp32_manager.loop
            )
            try:
                future.result(timeout=5)
            except Exception as e:
                print(f"Error during ESP32 disconnect: {str(e)}")
            
            server_sock.close()
            print("Server stopped")

def main():
    try:
        # Create and start asyncio thread
        asyncio_thread = AsyncioThread()
        asyncio_thread.start()
        asyncio_thread.started.wait()
        
        # Initialize ESP32 manager
        esp32_mac = "08:B6:1F:B9:36:76"  # Your ESP32's MAC address
        esp32_manager = ESP32BLEManager(esp32_mac)
        esp32_manager.loop = asyncio_thread.loop
        
        # Start ESP32 connection attempts in background
        esp32_manager.start_connection_thread(asyncio_thread.loop)
        
        # Initialize other components
        detector = SeizureDetector('seizure_detection_model.keras')
        data_reader = EEGDataReader(patient_num=1)
        server = SeizureAlertServer(detector, data_reader, esp32_manager)
        
        # Start server
        server.start()
        
    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
    finally:
        # Stop asyncio loop
        if asyncio_thread and asyncio_thread.loop:
            asyncio_thread.loop.call_soon_threadsafe(asyncio_thread.loop.stop)

if __name__ == "__main__":
    main()
