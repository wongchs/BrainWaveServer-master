import asyncio
import re
import time
import socket
import json
from Config import pattern as p
from BluetoothManager import scan_ble_devices
from BrainWaveManager import connect_to_board, release_board, write_as_csv
from PortManager import find_device

def start_socket_server(board, host='0.0.0.0', port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
    except socket.error as e:
        print(f"Socket binding failed: {e}")
        return

    server_socket.listen(1)

    print(f"Waiting for connection on {host}:{port}...")
    try:
        client_socket, client_address = server_socket.accept()
        print(f"Accepted connection from {client_address}")

        while True:
            data = board.get_current_board_data(20)  # Get latest data points
            if data is not None and data.size > 0:
                json_data = json.dumps(data.tolist())
                client_socket.send(json_data.encode())
            time.sleep(0.1)  # Adjust the delay as needed
    except Exception as e:
        print(f"Error in socket server: {e}")
    finally:
        client_socket.close()
        server_socket.close()

def start_bluetooth_server(board):
    server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    port = 6  # You can choose any port between 1 and 30
    try:
        server_sock.bind(("00:1A:7D:DA:71:13", port))
    except socket.error as e:
        print(f"Bluetooth socket binding failed: {e}")
        return

    server_sock.listen(1)

    print("Waiting for Bluetooth connection...")
    try:
        client_sock, client_info = server_sock.accept()
        print(f"Accepted connection from {client_info}")

        while True:
            data = board.get_current_board_data(20)  # Get latest data points
            if data is not None and data.size > 0:
                json_data = json.dumps(data.tolist())
                client_sock.send(json_data.encode())
            time.sleep(0.1)  # Adjust the delay as needed
    except Exception as e:
        print(f"Error in Bluetooth server: {e}")
    finally:
        client_sock.close()
        server_sock.close()

def main():
    port_list = find_device()
    serial_port = None
    if len(port_list) > 1:
        print("Selected the index: ")
        print(port_list)
        index = int(input())
        serial_port = port_list[index]
    elif len(port_list) == 1:
        serial_port = port_list[0]
    else:
        print("No dongle are found!")
        return
    
    devices = asyncio.run(scan_ble_devices())
    if devices and serial_port is not None:
        serial_number = None
        mac_address = None
        for device in devices:
            device_name = device[0]
            if p.search(device_name) is not None:
                serial_number = re.findall(r'\d+', device_name)
                serial_number = ''.join(serial_number)
                mac_address = device[1]

        if serial_number is not None and mac_address is not None:
            print("Connecting board...")
            board = None
            try:
                board = connect_to_board(mac_address, serial_number, serial_port)
                board.start_stream()
                print("Board connected and streaming. Starting Bluetooth server...")
                start_socket_server(board)
            except Exception as e:
                print(f"Error: {e}. Please ensure Bluetooth is available and enabled on your device.")
            finally:
                if board is not None:
                    data = board.get_current_board_data(20)
                    release_board(board)
                    if data is not None and data.size > 0:
                        write_as_csv(data)
                    else:
                        print("No data available to write to CSV.")
    else:
        print("No devices are found!")

if __name__ == "__main__":
    main()