import asyncio
from bleak import BleakScanner
import bluetooth
import time
import json


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


def start_bluetooth_server(board):
    port = 4
    uuid = "00001101-0000-1000-8000-00805F9B34FB"
    phone_mac_address = "00:B5:D0:BB:45:30"
    
    try:
        print(f"Searching for service on {phone_mac_address}...")
        service_matches = bluetooth.find_service(uuid=uuid, address=phone_mac_address)
        
        if len(service_matches) == 0:
            print("Couldn't find the specified service.")
            return

        first_match = service_matches[0]
        port = first_match["port"]
        name = first_match["name"]
        host = first_match["host"]

        print(f"Connecting to \"{name}\" on {host}")

        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((host, port))
        
        data = board.get_current_board_data(20)
        if data is not None and data.size > 0:
            json_data = json.dumps(data.tolist())
            try:
                sock.send(json_data.encode())
                
                time.sleep(5)

                print("Closing connection")
                sock.close()
            except Exception as e:
                print(f"Error sending data: {e}")
        
    except bluetooth.BluetoothError as e:
        print(f"Bluetooth Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
