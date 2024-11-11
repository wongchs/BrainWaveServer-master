import bluetooth
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import time

# MAC address of your ESP32
address = '08:B6:1F:B9:36:76'  # replace with your ESP32's MAC address
port = 1
seizure_onset = 780
seizure_happening = False
received_hello = False  # Flag to track if "hello" is received

# Search for and connect to the ESP32
nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True, flush_cache=True, lookup_class=False)
sock = None
for addr, name in nearby_devices:
    if address == addr:
        print("Found the device!")
        connected = False
        while not connected:
            try:
                sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                sock.connect((addr, port))
                connected = True
                break
            except bluetooth.btcommon.BluetoothError as e:
                print(f"Error connecting to device: {e}. Retrying in 5 seconds...")
                time.sleep(5)

if sock is None:
    print("Could not connect to the ESP32 device.")
    exit()

# Function to handle Bluetooth communication
def bluetooth_thread():
    global seizure_happening, received_hello
    while True:
        try:
            data = sock.recv(1024)
            if data:
                line = data.decode('utf-8').rstrip()
                if line == "hello":
                    print("Received hello from ESP32")
                    received_hello = True  # Set the flag to True
        except bluetooth.btcommon.BluetoothError as e:
            print(f"Error receiving data: {e}")
            break

# Function to handle seizure detection message sending
def seizure_notification_thread():
    global seizure_happening
    while True:
        if seizure_happening:
            try:
                sock.send("seizure detected\n".encode('utf-8'))  # Send seizure detected message
                print("Seizure detected!")
                time.sleep(1)  # Adjust the interval as needed
            except bluetooth.btcommon.BluetoothError as e:
                print(f"Error sending data: {e}")
                break
        else:
            time.sleep(0.1)  # Sleep to prevent busy-waiting

# Function to animate the graph
def func(frame, line):
    global seizure_happening
    t = np.arange(0, frame, 1)
    # Check if the time has reached the seizure onset
    if t[-1] >= seizure_onset:
        # Simulate an epileptic seizure by increasing the frequency and amplitude
        y = 3 * np.sin(10 * t)
        seizure_happening = True  # Set the flag to True
        time.sleep(0.1)  # Sleep to prevent busy-waiting
        seizure_happening = False  # Reset the flag
        
    else:
        y = np.sin(t)
    line.set_data(t, y)
    return line,

# Create and start the Bluetooth communication thread
bluetooth_thread_instance = threading.Thread(target=bluetooth_thread)
bluetooth_thread_instance.daemon = True
bluetooth_thread_instance.start()

# Create and start the seizure notification thread
notification_thread = threading.Thread(target=seizure_notification_thread)
notification_thread.daemon = True
notification_thread.start()

# Wait until "hello" is received before starting the animation
while not received_hello:
    time.sleep(0.1)

# Create the graph animation
fig = plt.figure()
ax = plt.axes(xlim=(0, 300), ylim=(-3.2, 3.2))
line, = ax.plot([], [], lw=2)
ax.set_xlabel('time')
ax.set_ylabel('millivolt (mV)')
ani = animation.FuncAnimation(fig, func, frames=np.arange(1, 840, 1), fargs=(line,), interval=10, blit=True)

# Show the graph animation
plt.show()

# Clean up
sock.close()