import asyncio
import re
from Config import pattern as p
from BluetoothManager import scan_ble_devices, start_bluetooth_server
from BrainWaveManager import connect_to_board, release_board, write_as_csv
from PortManager import find_device

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
                start_bluetooth_server(board)
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