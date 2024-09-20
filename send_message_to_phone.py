import bluetooth
import time

def send_message_to_phone(phone_mac_address):
    port = 4  # RFCOMM port
    uuid = "00001101-0000-1000-8000-00805F9B34FB"  # Standard SerialPortService ID

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

        print("Connected. Sending message...")
        message = "Hello, Android!"
        sock.send(message.encode())
        print(f"Sent message: {message}")

        # Keep the connection open for a while
        time.sleep(5)

        print("Closing connection")
        sock.close()

    except bluetooth.BluetoothError as e:
        print(f"Bluetooth Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    phone_mac_address = "00:B5:D0:BB:45:30"  # Replace with your Android device's Bluetooth MAC address
    send_message_to_phone(phone_mac_address)