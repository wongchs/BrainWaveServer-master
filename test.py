import asyncio
import json
import time
from datetime import datetime
import scipy.io
import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram
import os
import bluetooth
import threading

class SeizureDetector:
    def __init__(self, model_path='seizure_detection_model.keras'):
        print("Loading seizure detection model...")
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = 0.5  # Classification threshold
        print("Model loaded successfully")
        
    def preprocess_data(self, eeg_data):
        """Preprocess EEG data chunk exactly as done during training"""
        # Create spectrogram with same parameters as training
        f, t, Sxx = spectrogram(eeg_data, fs=5000, return_onesided=False)
        # Log transform and normalize
        SS = np.log1p(Sxx)
        arr = SS[:] / np.max(SS)
        # Reshape to match model input shape (1, 256, 22, 1)
        arr = arr.reshape(1, 256, 22, 1)
        return arr
        
    def predict(self, eeg_data):
        """Make seizure prediction on EEG data chunk"""
        processed_data = self.preprocess_data(eeg_data)
        prediction = self.model.predict(processed_data, verbose=0)
        # Return probability of seizure class
        return float(prediction[0][1])

class EEGDataReader:
    def __init__(self, patient_num, segment_type='preictal', segment_num=1):
        self.base_path = f'./kaggle/input/seizure-prediction/Patient_{patient_num}/Patient_{patient_num}/'
        self.file_path = os.path.join(
            self.base_path, 
            f'Patient_{patient_num}_{segment_type}_segment_{str(segment_num).zfill(4)}.mat'
        )
        self.current_position = 0
        self.load_data()
        
    def load_data(self):
        """Load EEG data from .mat file"""
        print(f"Loading EEG data from {self.file_path}")
        mat_data = scipy.io.loadmat(self.file_path)
        key = list(mat_data.keys())[-1]  # Get the last key containing data
        self.data = mat_data[key][0][0][0]
        print("EEG data loaded successfully")
        
    def get_chunk(self, chunk_size=5000):
        """Get next chunk of EEG data (1 second at 5000 Hz)"""
        if self.current_position + chunk_size > len(self.data[0]):
            self.current_position = 0
            
        chunk = self.data[0][self.current_position:self.current_position + chunk_size]
        self.current_position += chunk_size
        return chunk

class SeizureAlertServer:
    def __init__(self, detector, data_reader):
        self.detector = detector
        self.data_reader = data_reader
        
    def handle_client(self, client_sock, client_info):
        """Handle individual client connection with real-time seizure detection"""
        print(f"Handling connection from {client_info}")
        
        try:
            while True:
                # Get 1 second of EEG data (5000 samples at 5000 Hz)
                data_chunk = self.data_reader.get_chunk()
                
                # Get seizure probability from model
                seizure_prob = self.detector.predict(data_chunk)
                
                # Prepare data for sending
                alert_data = {
                    "data": data_chunk[:10].tolist(),  # Send first 10 samples as preview
                    "seizure_probability": seizure_prob,
                    "seizure_detected": bool(seizure_prob > self.detector.threshold),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Convert to JSON
                json_data = json.dumps(alert_data)
                
                # Send results to client
                client_sock.send(json_data.encode())
                
                # Print data to console
                print(f"Data sent to {client_info}: {json_data}")
                
                # If seizure detected, print additional warning
                if alert_data["seizure_detected"]:
                    print(f"⚠️ SEIZURE ALERT! Probability: {seizure_prob:.3f}")
                
                # Small delay to prevent overwhelming the connection
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error handling client {client_info}: {str(e)}")
        finally:
            client_sock.close()
            print(f"Connection closed with {client_info}")
    
    def start(self):
        """Start the Bluetooth server"""
        server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        port = bluetooth.PORT_ANY
        server_sock.bind(("", port))
        server_sock.listen(5)

        # Set up Bluetooth service
        uuid = "00001101-0000-1000-8000-00805F9B34FB"
        bluetooth.advertise_service(
            server_sock, 
            "SeizureAlertServer",
            service_id=uuid,
            service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
            profiles=[bluetooth.SERIAL_PORT_PROFILE]
        )

        local_address = bluetooth.read_local_bdaddr()
        print(f"Server started on Bluetooth address: {local_address}")
        print(f"Waiting for connections on RFCOMM channel {server_sock.getsockname()[1]}")

        try:
            while True:
                client_sock, client_info = server_sock.accept()
                print(f"Accepted connection from {client_info}")
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_sock, client_info)
                )
                client_thread.start()
        except KeyboardInterrupt:
            print("\nServer stopped by user")
        finally:
            server_sock.close()
            print("Server stopped")

def main():
    # Initialize components
    detector = SeizureDetector('seizure_detection_model.keras')
    data_reader = EEGDataReader(patient_num=1)  # Using Patient 1's data
    server = SeizureAlertServer(detector, data_reader)
    
    # Start server
    server.start()

if __name__ == "__main__":
    main()