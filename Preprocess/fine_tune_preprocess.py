
import os
import numpy as np
import pandas as pd

def read_waveform_csv(csv_path):
    """
    Reads a waveform CSV file and extracts non-zero, valid numeric data.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        np.ndarray: A NumPy array of non-zero, valid numeric data points of type float32.
    """
    try:
        # Load the CSV file, skipping the first two rows (metadata and NaNs)
        df = pd.read_csv(csv_path, skiprows=2, header=None)
        # Flatten the DataFrame to a 1D array
        data = df.values.flatten()
        # Remove NaNs and zeros
        data = data[~np.isnan(data)]  # Remove NaNs
        data = data[data != 0]        # Remove zeros
        # Convert to float32 for consistency
        data = data.astype(np.float64)
        return data
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return np.array([], dtype=np.float32)

def get_channel_data(channel_dir):
    """
    Reads all CSV files in a channel directory, concatenates the data, and returns a NumPy array.
    """
    csv_files = [os.path.join(channel_dir, f) for f in os.listdir(channel_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {channel_dir}")
        return np.array([], dtype=np.float32)
    data_list = [read_waveform_csv(csv_file) for csv_file in csv_files]
    if not data_list:
        print(f"No data found in {channel_dir}")
        return np.array([], dtype=np.float32)
    data_array = np.concatenate(data_list)
    return data_array

def create_dataset(base_dir, signal_length=25000):
    conditions = ['normal', 'misaligned', 'imbalanced', 'bearing fault']
    condition_labels = {'normal': 0, 'misaligned': 1, 'imbalanced': 2, 'bearing fault': 3}

    X_data = []
    y_data = []

    channels = ['ch1', 'ch2', 'ch3', 'ch4']
    
    visit = 0
    
    for condition in conditions:
        condition_dir = os.path.join(base_dir, condition)
        label = condition_labels[condition]

        if not os.path.isdir(condition_dir):
            print(f"Condition directory {condition_dir} does not exist.")
            continue

        # Traverse speed subdirectories
        speed_dirs = [d for d in os.listdir(condition_dir) if os.path.isdir(os.path.join(condition_dir, d))]
        for speed in speed_dirs:
            speed_dir = os.path.join(condition_dir, speed, 'waveformData')
            if not os.path.isdir(speed_dir):
                print(f"Waveform data directory {speed_dir} does not exist.")
                continue

            # Collect data from all channels
            channel_data_arrays = []
            for ch in channels:
                ch_dir = os.path.join(speed_dir, ch)
                if not os.path.isdir(ch_dir):
                    print(f"Channel directory {ch_dir} does not exist.")
                    break  # Cannot proceed without all channels

                # Get the data for the channel
                ch_data = get_channel_data(ch_dir)
                if len(ch_data) == 0:
                    print(f"No data found for channel {ch} in {speed_dir}")
                    break  # Cannot proceed without all channels
                channel_data_arrays.append(ch_data)
                
                print(f'\n {condition} & {speed} from channle {ch} data collected. ')
                visit = visit +1 
                print(f'\nCSV file visited {visit}')
                print(f'Length of the data in {ch} is {len(ch_data)}')
                
            else:
                # Proceed if all channels have data
                # Find the minimum length across all channels
                min_length = min(len(ch_data) for ch_data in channel_data_arrays)
                print(f'\n----in the else block:--- The min_length of the chanle is {min_length}')
                # Truncate all channels to the minimum length
                channel_data_arrays = [ch_data[:min_length] for ch_data in channel_data_arrays]
                # Determine the number of samples
                num_samples = min_length // signal_length
                if num_samples == 0:
                    print(f"Not enough data to create samples for {speed_dir}")
                    continue
                # Truncate data arrays to total_length = num_samples * signal_length
                total_length = num_samples * signal_length
                channel_data_arrays = [ch_data[:total_length] for ch_data in channel_data_arrays]
                # Reshape data arrays into (num_samples, signal_length)
                channel_data_arrays = [ch_data.reshape(num_samples, signal_length) for ch_data in channel_data_arrays]
                print(f'\nChannel data array shape is {len(channel_data_arrays)}')
                # Stack channels to create samples
                # Shape of each ch_data is (num_samples, signal_length)
                # We need to stack along a new axis to get shape (num_samples, num_channels, signal_length)
                samples = np.stack(channel_data_arrays, axis=1)  # Shape: (num_samples, num_channels, signal_length)
                print(f'\nShape of the sample is  {samples.shape}')
                X_data.append(samples)
                y_data.extend([label] * num_samples)
    # Concatenate all samples
    if X_data:
        X = np.concatenate(X_data, axis=0)
        y = np.array(y_data, dtype=np.int64)
    else:
        X = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.int64)
    print(f"\nFinal X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Final y shape: {y.shape}, dtype: {y.dtype}")

    return X, y

def main():
    base_dir = '/home/dapgrad/tenzinl2/TFPred/raw_data'  # Replace with your actual path
    signal_length = 1013 #25000  # Desired signal length
    X, y = create_dataset(base_dir, signal_length)

    print(f'\nFirst element of my data is {X[0][0][0]}')
    
    # Example code to compare with existing data (if available)
    X_data = np.load(os.path.join("/home/dapgrad/tenzinl2/TFPred/data", 'X_test.npy'), allow_pickle=True)  # Shape: (num_samples, channels, signal_length)
    y_data = np.load(os.path.join("/home/dapgrad/tenzinl2/TFPred/data", 'y_test.npy'), allow_pickle=True)
    
    print(f'\nFirst element of their data is {X_data[0][0][0]}')
    print(f"\nFinal X_data shape: {X_data.shape}, dtype: {X_data.dtype}")
    print(f"Final y_data shape: {y_data.shape}, dtype: {y_data.dtype}")
    
    # Save the datasets
    np.save('X_t.npy', X)
    np.save('y_t.npy', y)

if __name__ == "__main__":
    main()
