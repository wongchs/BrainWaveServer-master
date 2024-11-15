import asyncio
import json
import time
import random
import threading
import bluetooth
from datetime import datetime
import scipy.io
import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram
import os

class MatFileReader:
    def __init__(self, patient_num, segment_type, segment_num):
        self.base_path = f'./kaggle/input/seizure-prediction/Patient_{patient_num}/Patient_{patient_num}/'
        self.current_file = f'Patient_{patient_num}_{segment_type}_segment_{str(segment_num).zfill(4)}.mat'
        self.current_position = 0
        self.data = None
        self.load_data()
        
    def load_data(self):
        file_path = os.path.join(self.base_path, self.current_file)
        mat_data = scipy.io.loadmat(file_path)
        key = list(mat_data.keys())[-1]  # Get the last key which contains the data
        self.data = mat_data[key][0][0][0]  # Access the EEG data
        
    def get_next_chunk(self, chunk_size=5000):
        if self.current_position + chunk_size > len(self.data[0]):
            self.current_position = 0
        
        chunk = self.data[0][self.current_position:self.current_position + chunk_size]
        self.current_position += chunk_size
        return chunk

def process_eeg_data(data_chunk):
    # Create spectrogram
    f, t, Sxx = spectrogram(data_chunk, fs=5000, return_onesided=False)
    SS = np.log1p(Sxx)
    arr = SS[:] / np.max(SS)
    
    # Reshape for model input
    arr = arr.reshape(1, 256, 22, 1)
    return arr

def handle_client(client_sock, client_info, mat_reader, model):
    print(f"Handling connection from {client_info}")
    
    try:
        last_seizure_time = time.time() - 10
        while True:
            # Get chunk of EEG data from .mat file
            data_chunk = mat_reader.get_next_chunk()
            
            # Convert to list for JSON serialization
            data_list = data_chunk[:10].tolist()  # Send first 10 samples like original code
            
            current_time = time.time()
            if current_time - last_seizure_time >= 10:
                # Process data for seizure detection
                processed_data = process_eeg_data(data_chunk)
                
                # Make prediction using the model
                prediction = model.predict(processed_data, verbose=0)
                seizure_detected = bool(prediction[0][1] > 0.5)  # Threshold of 0.5 for binary classification
                
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
            
            time.sleep(1)
            
    except Exception as e:
        print(f"Error in handle_client: {str(e)}")
    finally:
        client_sock.close()
        print(f"Connection closed with {client_info}")

def start_bluetooth_server():
    # Load the trained model
    model = tf.keras.models.load_model('seizure_detection_model.keras')
    
    # Initialize MatFileReader (you can modify parameters as needed)
    mat_reader = MatFileReader(patient_num=1, segment_type='preictal', segment_num=1)
    
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    port = bluetooth.PORT_ANY
    server_sock.bind(("", port))
    server_sock.listen(5)

    uuid = "00001101-0000-1000-8000-00805F9B34FB"
    bluetooth.advertise_service(server_sock, "BrainWaveServer",
                              service_id=uuid,
                              service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                              profiles=[bluetooth.SERIAL_PORT_PROFILE])

    local_address = bluetooth.read_local_bdaddr()
    print(f"Server started on Bluetooth address: {local_address}")
    print(f"Waiting for connections on RFCOMM channel {server_sock.getsockname()[1]}")

    try:
        while True:
            client_sock, client_info = server_sock.accept()
            print(f"Accepted connection from {client_info}")
            client_thread = threading.Thread(
                target=handle_client, 
                args=(client_sock, client_info, mat_reader, model)
            )
            client_thread.start()
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        server_sock.close()
        print("Server stopped")

if __name__ == "__main__":
    start_bluetooth_server()