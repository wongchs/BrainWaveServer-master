import socket

def bluetooth_chat_server():
    # Set your server's Bluetooth MAC address
    server_mac_address = "00:1A:7D:DA:71:13"  # Your server's Bluetooth MAC address
    # Set the target device (phone) Bluetooth MAC address
    phone_mac_address = "00:B5:D0:BB:45:30"
    port = 6  # RFCOMM port commonly used for Bluetooth connections

    # Create Bluetooth socket
    server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)

    # Bind the socket to the server's Bluetooth MAC address and RFCOMM port
    server_sock.bind((server_mac_address, port))
    server_sock.listen(1)

    print(f"Waiting for connection on RFCOMM channel {port}...")

    try:
        # Establish the connection to the phone (client)
        client_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        client_sock.connect((phone_mac_address, port))
        print(f"Connected to phone at {phone_mac_address}")

        # Send a message to the phone
        message = "Hello from server!"
        client_sock.send(message.encode('utf-8'))
        print(f"Message sent to {phone_mac_address}: {message}")

        # Optionally, wait for a response
        data = client_sock.recv(1024)
        if data:
            print(f"Received response from phone: {data.decode('utf-8')}")

    except OSError as e:
        print(f"Connection error: {e}")
    finally:
        print("Closing connection")
        client_sock.close()
        server_sock.close()

if __name__ == "__main__":
    bluetooth_chat_server()
