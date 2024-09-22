import asyncio
from bleak import BleakScanner
import bluetooth
import time
import json
import numpy as np
import threading

def handle_client(client_sock, client_info, board):
    print(f"Handling connection from {client_info}")
    try:
        while True:
            data = board.get_current_board_data(10)  # Get 10 data points
            if data is not None and data.size > 0:
                data_list = data[0, :10].tolist()
                json_data = json.dumps(data_list)
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


# directly send to phone via mac address
# def start_bluetooth_server(board):
#     port = 4
#     uuid = "00001101-0000-1000-8000-00805F9B34FB"
#     phone_mac_address = "00:B5:D0:BB:45:30"
    
#     try:
#         print(f"Searching for service on {phone_mac_address}...")
#         service_matches = bluetooth.find_service(uuid=uuid, address=phone_mac_address)
        
#         if len(service_matches) == 0:
#             print("Couldn't find the specified service.")
#             return

#         first_match = service_matches[0]
#         port = first_match["port"]
#         name = first_match["name"]
#         host = first_match["host"]

#         print(f"Connecting to \"{name}\" on {host}")

#         sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
#         sock.connect((host, port))
        
#         while True:
#             data = board.get_current_board_data(10)  # Get 10 data points
#             if data is not None and data.size > 0:
#                 # Convert data to a list of 10 points (assuming single channel)
#                 data_list = data[0, :10].tolist()
#                 json_data = json.dumps(data_list)
#                 try:
#                     sock.send(json_data.encode())
#                     print("Data sent:", json_data)
#                 except Exception as e:
#                     print(f"Error sending data: {e}")
#                     break
            
#             time.sleep(5)  # Wait for 5 seconds before sending next batch

#     except bluetooth.BluetoothError as e:
#         print(f"Bluetooth Error: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#     finally:
#         sock.close()
#         print("Connection closed")

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

