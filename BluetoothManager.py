import asyncio
from bleak import BleakClient, BleakScanner
import bluetooth
import time
import json
import numpy as np
import threading
from datetime import datetime
import random

class ESP32BLEManager:
    def __init__(self, esp32_mac):
        self.esp32_mac = esp32_mac
        self.client = None
        self.connected = False
        self.lock = threading.Lock()
        self.notification_characteristic = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        self.write_characteristic = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
        self.loop = None

    async def connect(self):
        """Establish BLE connection with ESP32"""
        try:
            print(f"Scanning for ESP32 device: {self.esp32_mac}")
            device = await BleakScanner.find_device_by_address(self.esp32_mac, timeout=20.0)
            
            if not device:
                print("ESP32 device not found")
                return False

            print(f"Found ESP32 device, attempting to connect...")
            self.client = BleakClient(device)
            await self.client.connect()
            self.connected = True
            print("Successfully connected to ESP32")
            return True

        except Exception as e:
            print(f"Connection error: {str(e)}")
            self.connected = False
            return False

    async def send_alert(self, message="SEIZURE_DETECTED\n"):
        """Send alert to ESP32"""
        try:
            if not self.connected or not self.client:
                print("Not connected to ESP32")
                return False

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

def handle_client(client_sock, client_info, board):
    print(f"Handling connection from {client_info}")
    
    # Initialize ESP32 manager
    esp32_mac = "08:B6:1F:B9:36:76"  # Your ESP32's MAC address
    esp32_manager = ESP32BLEManager(esp32_mac)
    
    # Create and start asyncio thread
    asyncio_thread = AsyncioThread()
    asyncio_thread.start()
    
    # Wait for the event loop to be ready
    asyncio_thread.started.wait()
    
    # Connect to ESP32
    try:
        future = asyncio.run_coroutine_threadsafe(
            esp32_manager.connect(), 
            asyncio_thread.loop
        )
        connected = future.result(timeout=30)  # Add timeout to prevent hanging
        
        if not connected:
            print("Warning: Could not establish initial connection to ESP32")
        
        last_seizure_time = time.time() - 10
        while True:
            data = board.get_current_board_data(10)
            if data is not None and data.size > 0:
                data_list = data[0, :10].tolist()
                current_time = time.time()
                
                if current_time - last_seizure_time >= 10:
                    seizure_detected = random.choice([True, False])
                    if seizure_detected:
                        last_seizure_time = current_time
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        json_data = json.dumps({
                            "data": data_list,
                            "seizure_detected": True,
                            "timestamp": timestamp
                        })
                        # Send alert to ESP32
                        future = asyncio.run_coroutine_threadsafe(
                            esp32_manager.send_alert(),
                            asyncio_thread.loop
                        )
                        future.result(timeout=5)  # Add timeout for alert sending
                    else:
                        json_data = json.dumps({"data": data_list})
                else:
                    json_data = json.dumps({"data": data_list})
                
                client_sock.send(json_data.encode())
                print(f"Data sent to {client_info}: {json_data}")

            time.sleep(1)
            
    except Exception as e:
        print(f"Error in handle_client: {str(e)}")
    finally:
        client_sock.close()
        # Disconnect from ESP32
        try:
            future = asyncio.run_coroutine_threadsafe(
                esp32_manager.disconnect(),
                asyncio_thread.loop
            )
            future.result(timeout=5)
        except Exception as e:
            print(f"Error during disconnect: {str(e)}")
            
        # Stop asyncio loop
        asyncio_thread.loop.call_soon_threadsafe(asyncio_thread.loop.stop)
        print(f"Connection closed with {client_info}")


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


# device as the main bluetooth server, accept connection from android clients
def start_bluetooth_server(board):
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    port = bluetooth.PORT_ANY
    server_sock.bind(("", port))  # Bind to all available Bluetooth adapters
    server_sock.listen(5)  # Allow up to 5 simultaneous connections

    uuid = "00001101-0000-1000-8000-00805F9B34FB"
    bluetooth.advertise_service(server_sock, "BrainWaveServer",
                                service_id=uuid,
                                service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                                profiles=[bluetooth.SERIAL_PORT_PROFILE])

    # Get and print the Bluetooth adapter's address for user reference
    local_address = bluetooth.read_local_bdaddr()
    print(f"Server started on Bluetooth address: {local_address}")
    print(f"Waiting for connections on RFCOMM channel {server_sock.getsockname()[1]}")

    try:
        while True:
            client_sock, client_info = server_sock.accept()
            print(f"Accepted connection from {client_info}")
            client_thread = threading.Thread(target=handle_client, args=(client_sock, client_info, board))
            client_thread.start()
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        server_sock.close()
        print("Server stopped")

