# bluetooth_server.py

import bluetooth

def start_bluetooth_server():
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    port = bluetooth.PORT_ANY
    server_sock.bind(("", port))
    server_sock.listen(1)

    port = server_sock.getsockname()[1]

    uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

    bluetooth.advertise_service(server_sock, "PythonBluetoothServer",
                                service_id=uuid,
                                service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                                profiles=[bluetooth.SERIAL_PORT_PROFILE])

    print(f"Waiting for connection on RFCOMM channel {port}")

    client_sock, client_info = server_sock.accept()
    print(f"Accepted connection from {client_info}")

    try:
        while True:
            data = client_sock.recv(1024)
            if not data:
                break
            print(f"Received: {data.decode('utf-8')}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_sock.close()
        server_sock.close()

if __name__ == "__main__":
    start_bluetooth_server()