import asyncio
from bleak import BleakScanner
import bluetooth
import time
import json
import numpy as np
import threading
from datetime import datetime
import random

def handle_client(client_sock, client_info, board):
    print(f"Handling connection from {client_info}")
    try:
        last_seizure_time = time.time() - 10
        while True:
            data = board.get_current_board_data(10)  # Get 10 data points
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
                    else:
                        json_data = json.dumps({"data": data_list})
                else:
                    json_data = json.dumps({"data": data_list})
                
                client_sock.send(json_data.encode())
                print(f"Data sent to {client_info}: {json_data}")

            time.sleep(1)  # Wait for 1 second before sending next batch
    except bluetooth.BluetoothError as e:
        print(f"Bluetooth Error with {client_info}: {e}")
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

