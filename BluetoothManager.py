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


def handle_client(client_sock, client_info, board, esp32_manager):
    print(f"Handling connection from {client_info}")
    
    try:
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
                        # Try to send alert to ESP32 if connected
                        if esp32_manager.connected:
                            future = asyncio.run_coroutine_threadsafe(
                                esp32_manager.send_alert(),
                                esp32_manager.loop
                            )
                            try:
                                future.result(timeout=5)
                            except Exception as e:
                                print(f"Failed to send alert: {str(e)}")
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
    # Initialize ESP32 manager
    esp32_mac = "08:B6:1F:B9:36:76"  # Your ESP32's MAC address
    
    # Create and start asyncio thread
    asyncio_thread = AsyncioThread()
    asyncio_thread.start()
    asyncio_thread.started.wait()
    
    # Initialize ESP32 manager with the asyncio loop
    esp32_manager = ESP32BLEManager(esp32_mac)
    esp32_manager.loop = asyncio_thread.loop
    
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
                args=(client_sock, client_info, board, esp32_manager)
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
