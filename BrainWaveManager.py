import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import numpy as np
import pandas as pd

def release_board(board):
    board.stop_stream()
    board.release_session()


def connect_to_board(mac_address, serial_number, serial_port, timeout=0, ip_port=0, ip_protocol=0, board_id=1, log=True):
    params = BrainFlowInputParams()
    params.ip_port = ip_port
    params.serial_port = serial_port
    params.mac_address = mac_address
    params.ip_protocol = ip_protocol
    params.serial_number = serial_number
    params.ip_protocol = ip_protocol
    params.timeout = timeout

    if log:
        BoardShim.enable_dev_board_logger()
    else:
        BoardShim.disable_board_logger()

    board = BoardShim(board_id, params)
    board.prepare_session()
    return board

def get_selected_channels_data(board, channels=[1, 2, 3], num_samples=20):
    """
    Get data from specified EEG channels
    
    Args:
        board: BoardShim instance
        channels: List of channel indices (0-based) to read from
        num_samples: Number of samples to get
    
    Returns:
        numpy array with selected channel data
    """
    data = board.get_current_board_data(num_samples)
    if data is not None and data.size > 0:
        # Get all EEG channels available on Ganglion
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value)
        # Select only the requested channels
        selected_channels = [eeg_channels[i] for i in channels if i < len(eeg_channels)]
        return data[selected_channels]
    return None


def write_as_csv(data):
    # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value)
    df = pd.DataFrame(np.transpose(data))
    print('Data From the Board')
    print(df.head(10))

    # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
    DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
    restored_data = DataFilter.read_file('test.csv')
    restored_df = pd.DataFrame(np.transpose(restored_data))
    print('Data From the File')
    print(restored_df.head(10))
