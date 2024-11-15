import json
import time
from datetime import datetime
import scipy.io
import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram
import os

class SeizureDetector:
    def __init__(self, model_path='seizure_detection_model.keras'):
        print("Loading seizure detection model...")
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = 0.8
        self.prediction_window = []
        self.window_size = 10
        print("Model loaded successfully")
        
    def preprocess_data(self, eeg_data):
        """Preprocess EEG data chunk exactly as done during training"""
        try:
            # Ensure the data is the right shape and type
            eeg_data = np.array(eeg_data, dtype=np.float32)
            if len(eeg_data.shape) > 1:
                eeg_data = eeg_data.flatten()
            
            # Basic normalization
            eeg_data = (eeg_data - np.mean(eeg_data)) / (np.std(eeg_data) + 1e-8)
            
            # Calculate spectrogram with fixed parameters
            nperseg = 256
            noverlap = nperseg // 2
            
            f, t, Sxx = spectrogram(
                eeg_data,
                fs=5000,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=256,
                window='hann',
                mode='magnitude'
            )
            
            print(f"Spectrogram shape before processing: {Sxx.shape}")
            
            # Ensure we have correct number of frequency bins
            if Sxx.shape[0] != 22:
                # Create frequency bins using actual frequency values
                target_freqs = np.linspace(f[0], f[-1], 22)
                new_Sxx = np.zeros((22, Sxx.shape[1]))
                
                # For each target frequency bin
                for i in range(22):
                    if i == 21:  # Handle the last bin separately
                        freq_mask = f >= target_freqs[i]
                    else:
                        freq_mask = (f >= target_freqs[i]) & (f < target_freqs[i + 1])
                    
                    if np.any(freq_mask):
                        new_Sxx[i] = np.mean(Sxx[freq_mask], axis=0)
                    else:
                        # If no frequencies fall in this bin, use nearest neighbor
                        nearest_idx = np.abs(f - target_freqs[i]).argmin()
                        new_Sxx[i] = Sxx[nearest_idx]
                
                Sxx = new_Sxx
            
            # Ensure we have correct number of time bins
            if Sxx.shape[1] != 256:
                from scipy.ndimage import zoom
                zoom_factor = (1, 256/Sxx.shape[1])
                Sxx = zoom(Sxx, zoom_factor, order=1)
            
            print(f"Spectrogram shape after processing: {Sxx.shape}")
            
            # Log transform and normalize
            Sxx = np.log1p(np.abs(Sxx))
            Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx) + 1e-8)
            
            # Reshape for model input
            arr = Sxx.T.reshape(1, 256, 22, 1)
            print(f"Final array shape: {arr.shape}")
            
            return arr
            
        except Exception as e:
            print(f"Error in preprocess_data: {str(e)}")
            print(f"EEG data shape: {eeg_data.shape}")
            print(f"EEG data type: {eeg_data.dtype}")
            raise
        
    def predict(self, eeg_data):
        """Make seizure prediction on EEG data chunk"""
        try:
            processed_data = self.preprocess_data(eeg_data)
            prediction = self.model.predict(processed_data, verbose=0)
            prob = float(prediction[0][1])
            
            # Update moving average window
            self.prediction_window.append(prob)
            if len(self.prediction_window) > self.window_size:
                self.prediction_window.pop(0)
            
            # Return average probability over window
            return np.mean(self.prediction_window)
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            raise
        
        
class EEGDataReader:
    def __init__(self, patient_num, segment_type='preictal', segment_num=1):
        # self.base_path = f'./kaggle/input/seizure-prediction/Patient_{patient_num}/Patient_{patient_num}/'
        # self.file_path = os.path.join(
        #     self.base_path, 
        #     f'Patient_{patient_num}_{segment_type}_segment_{str(segment_num).zfill(4)}.mat'
        # )
        self.base_path = f'./kaggle/input/seizure-prediction/Dog_{patient_num}/Dog_{patient_num}/'
        self.file_path = os.path.join(
            self.base_path, 
            f'Dog_{patient_num}_{segment_type}_segment_{str(segment_num).zfill(4)}.mat'
        )
        self.current_position = 0
        self.load_data()
        
    def load_data(self):
        """Load EEG data from .mat file"""
        try:
            print(f"Loading EEG data from {self.file_path}")
            mat_data = scipy.io.loadmat(self.file_path)
            key = list(mat_data.keys())[-1]
            self.data = mat_data[key][0][0][0]
            print(f"EEG data loaded successfully. Shape: {self.data.shape}")
            
            # Data quality checks
            if np.any(np.isnan(self.data)):
                print("Warning: NaN values found in EEG data")
            if np.any(np.isinf(self.data)):
                print("Warning: Infinite values found in EEG data")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
    def get_chunk(self, chunk_size=5000):
        """Get next chunk of EEG data (1 second at 5000 Hz)"""
        try:
            if self.current_position + chunk_size > len(self.data[0]):
                self.current_position = 0
                
            chunk = self.data[0][self.current_position:self.current_position + chunk_size]
            self.current_position += chunk_size
            return chunk
        except Exception as e:
            print(f"Error getting chunk: {str(e)}")
            raise

class SeizureMonitor:
    def __init__(self, detector, data_reader):
        self.detector = detector
        self.data_reader = data_reader
        self.alert_history = []
        self.consecutive_detections = 0
        self.min_consecutive_detections = 3
        
    def log_alert(self, alert_data):
        """Log alert data to a file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"seizure_alerts_{timestamp[:8]}.json"
            
            self.alert_history.append(alert_data)
            
            with open(filename, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            print(f"Error logging alert: {str(e)}")
            
    def start_monitoring(self):
        """Start continuous seizure monitoring"""
        print("Starting seizure monitoring...")
        print(f"Using threshold: {self.detector.threshold}")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                try:
                    data_chunk = self.data_reader.get_chunk()
                    seizure_prob = self.detector.predict(data_chunk)
                    
                    is_seizure = seizure_prob > self.detector.threshold
                    
                    if is_seizure:
                        self.consecutive_detections += 1
                    else:
                        self.consecutive_detections = 0
                    
                    true_detection = self.consecutive_detections >= self.min_consecutive_detections
                    
                    alert_data = {
                        "data_preview": data_chunk[:10].tolist(),
                        "seizure_probability": float(seizure_prob),  # Ensure JSON serializable
                        "seizure_detected": true_detection,
                        "consecutive_detections": self.consecutive_detections,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    self.log_alert(alert_data)
                    
                    print(f"\rCurrent seizure probability: {seizure_prob:.3f} "
                          f"(Consecutive detections: {self.consecutive_detections})", end="")
                    
                    if true_detection:
                        print(f"\n⚠️ SEIZURE ALERT! Probability: {seizure_prob:.3f}")
                        print(f"Timestamp: {alert_data['timestamp']}")
                        print("Data preview:", alert_data["data_preview"])
                        print("-" * 50)
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"\nError in monitoring loop: {str(e)}")
                    print("Continuing to next iteration...")
                    time.sleep(1)  # Add delay before retry
                    
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            print("\nMonitoring ended")
            print(f"Total alerts logged: {len(self.alert_history)}")

def main():
    try:
        detector = SeizureDetector('seizure_detection_model.keras')
        data_reader = EEGDataReader(patient_num=5)
        monitor = SeizureMonitor(detector, data_reader)
        monitor.start_monitoring()
    except Exception as e:
        print(f"Fatal error in main: {str(e)}")

if __name__ == "__main__":
    main()