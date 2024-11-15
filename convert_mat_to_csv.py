import scipy.io
import pandas as pd
import os
import numpy as np

def convert_mat_to_csv(patient_num, types, num_segments, output_dir='./csv_output'):
    """
    Convert .mat files containing EEG data to CSV format.
    
    Parameters:
    -----------
    patient_num : int
        Patient number (e.g., 1 or 2)
    types : list
        List of segment types (e.g., ['Patient_1_interictal_segment', 'Patient_1_preictal_segment'])
    num_segments : int
        Number of segments to process
    output_dir : str
        Directory where CSV files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    base_path = f'./kaggle/input/seizure-prediction/Patient_{patient_num}/Patient_{patient_num}/'
    
    for typ in types:
        for j in range(num_segments):
            # Construct file paths
            mat_file = os.path.join(base_path, f'{typ}_{str(j + 1).zfill(4)}.mat')
            
            try:
                # Load .mat file
                data = scipy.io.loadmat(mat_file)
                
                # Extract the key and data
                k = typ.replace(f'Patient_{patient_num}_', '') + '_'
                d_array = data[k + str(j + 1)][0][0][0]
                
                # Convert to DataFrame
                # Each row represents a time point, each column represents a channel
                df = pd.DataFrame(d_array.T)  # Transpose to have channels as columns
                
                # Add metadata columns
                df['segment_type'] = typ.split('_')[2]  # 'interictal' or 'preictal'
                df['segment_number'] = j + 1
                df['patient_number'] = patient_num
                
                # Save to CSV
                output_file = os.path.join(output_dir, f'patient_{patient_num}_{typ}_{j + 1}.csv')
                df.to_csv(output_file, index=True, index_label='time_point')
                
                print(f"Converted {mat_file} to {output_file}")
                
                # Print basic statistics
                print(f"Shape: {df.shape}")
                print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
                print("---")
                
            except Exception as e:
                print(f"Error processing {mat_file}: {str(e)}")
                continue

# Example usage
def main():
    # Convert data for Patient 1
    types_p1 = ['Patient_1_interictal_segment', 'Patient_1_preictal_segment']
    convert_mat_to_csv(1, types_p1, 18)
    
    # Convert data for Patient 2
    types_p2 = ['Patient_2_interictal_segment', 'Patient_2_preictal_segment']
    convert_mat_to_csv(2, types_p2, 18)

if __name__ == "__main__":
    main()