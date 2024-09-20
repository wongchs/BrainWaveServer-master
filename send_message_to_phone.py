import socket

def send_message_to_phone(phone_mac_address):
    port = 4  # Make sure this matches the port in your Kotlin app
    client_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    try:
        print(f"Connecting to phone at {phone_mac_address} on port {port}...")
        client_sock.connect((phone_mac_address, port))
        print("Connection established!")
        
        message = "Hi"
        client_sock.send(message.encode())
        print(f"Sent message: {message}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_sock.close()

if __name__ == "__main__":
    phone_mac_address = "00:B5:D0:BB:45:30"
    send_message_to_phone(phone_mac_address)