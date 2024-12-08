import os
import numpy as np
import pandas as pd
import torch

# These are placeholders adapted from the original code.
def _normalization(data, normalization_type):
    if normalization_type == "0-1":
        data_min = data.min()
        data_max = data.max()
        if data_max != data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)
    return data

def _transformation(sub_data, backbone):
    if backbone == "ResNet1D":
        # If needed, reshape sub_data to (1, length). 
        # Here sub_data is (signal_length,) for single channel.
        sub_data = sub_data[np.newaxis, :]
    else:
        raise NotImplementedError(f"Model {backbone} is not implemented.")
    return sub_data

def read_file(filepath):
    """
    Reads a waveform CSV file, ignoring the first two rows, removing NaNs and zeros.
    Returns a 1D numpy array of numeric data.
    """
    try:
        df = pd.read_csv(filepath, skiprows=2, header=None)
        data = df.values.flatten()
        # Remove NaNs and zeros
        data = data[~np.isnan(data)]
        data = data[data != 0]
        data = data.astype(np.float32)  # use float32 for consistency
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return np.array([], dtype=np.float32)

def PU(datadir, load, data_length, labels, window, normalization, backbone, number):
    """
    A function similar to the original PU function.
    It returns a dataset dictionary {label: np.array([...])} of shape (num_samples, 1, data_length)
    after reading and processing data from multiple channels and averaging them into a single channel.

    Args:
        datadir (str): Base directory where data is stored.
        load (int): Unused in this version. Previously used for selecting working condition, 
                    here you can ignore or adapt if needed.
        data_length (int): Length of each segment (equivalent to signal_length).
        labels (list): List of label indices.
        window (int): Unused in this adapted code. 
                      If you need segmentation by window, integrate it accordingly.
        normalization (str): Normalization type ("0-1" supported).
        backbone (str): Model backbone name ("ResNet1D" supported).
        number (int): Total number of samples needed. (num_train+num_validation+num_test)

    Returns:
        dict: {label: np.array([...], dtype=float32)} with shape (num_samples, 1, data_length).
    """

    # Conditions and mapping: adjust these if needed
    condition_map = {0: 'normal', 1: 'misaligned', 2: 'imbalanced', 3: 'bearing fault'}

    channels = ['ch1', 'ch2', 'ch3', 'ch4']

    dataset = {lbl: [] for lbl in labels}

    for lbl in labels:
        condition = condition_map[lbl]
        condition_dir = os.path.join(datadir, condition)

        if not os.path.isdir(condition_dir):
            continue

        # Traverse speed subdirectories
        speed_dirs = [d for d in os.listdir(condition_dir) if os.path.isdir(os.path.join(condition_dir, d))]
        for speed in speed_dirs:
            speed_dir = os.path.join(condition_dir, speed, 'waveformData')
            if not os.path.isdir(speed_dir):
                continue

            # Gather data from all channels
            channel_data_arrays = []
            valid = True
            for ch in channels:
                ch_dir = os.path.join(speed_dir, ch)
                if not os.path.isdir(ch_dir):
                    valid = False
                    break
                # Read all CSV files in ch_dir and concatenate
                csv_files = [os.path.join(ch_dir, f) for f in os.listdir(ch_dir) if f.endswith('.csv')]
                if not csv_files:
                    valid = False
                    break

                ch_data_list = []
                for csv_file in csv_files:
                    ch_data = read_file(csv_file)
                    if len(ch_data) > 0:
                        ch_data_list.append(ch_data)
                if not ch_data_list:
                    valid = False
                    break
                # Concatenate all data from this channel
                ch_data_array = np.concatenate(ch_data_list)
                channel_data_arrays.append(ch_data_array)

            if not valid or not channel_data_arrays:
                continue

            # Ensure all channels have the same length
            min_length = min(len(ch_data) for ch_data in channel_data_arrays)
            channel_data_arrays = [ch_data[:min_length] for ch_data in channel_data_arrays]

            # Determine how many samples we can form
            # Originally, you might integrate `number` and `window` logic here.
            # For simplicity, just use data_length for segmentation:
            num_samples = min_length // data_length
            if num_samples == 0:
                continue

            total_length = num_samples * data_length
            channel_data_arrays = [ch_data[:total_length] for ch_data in channel_data_arrays]

            # Reshape each channel to (num_samples, data_length)
            channel_data_arrays = [ch_data.reshape(num_samples, data_length) for ch_data in channel_data_arrays]

            # Stack channels: shape (num_samples, 4, data_length)
            samples = np.stack(channel_data_arrays, axis=1).astype(np.float32)

            # Average channels to get single channel: (num_samples, 1, data_length)
            samples = np.mean(samples, axis=1, keepdims=True)

            # Apply normalization if requested
            # Normalization per sample or per entire dataset could differ,
            # here we assume per-sample normalization if needed.
            # Loop over samples and normalize each? Or normalize after concatenation?
            # Simpler: just normalize channel-wise if needed:
            if normalization == "0-1":
                # Flatten, normalize, then reshape
                flat = samples.reshape(-1)
                flat = _normalization(flat, normalization)
                samples = flat.reshape(num_samples, 1, data_length)

            # Apply transformation (e.g., ResNet1D expects (1, data_length))
            # Already single channel: (num_samples, 1, data_length) is fine.
            # If needed, apply _transformation on each sample:
            transformed_samples = []
            for i in range(num_samples):
                transformed = _transformation(samples[i,0,:], backbone)  # shape (1, data_length)
                transformed_samples.append(transformed[np.newaxis,:,:]) # add batch dim temporarily
            transformed_samples = np.concatenate(transformed_samples, axis=0) # (num_samples, 1, data_length)

            dataset[lbl].append(transformed_samples)

    # Concatenate all data for each label
    for lbl in dataset:
        if dataset[lbl]:
            dataset[lbl] = np.concatenate(dataset[lbl], axis=0)
        else:
            dataset[lbl] = np.array([], dtype=np.float32)

    return dataset

def PUloader(args):
    """
    A loader function similar to the original PUloader.
    It parses args, calls PU, and returns the dataset dictionary.
    """
    label_set_list = list(int(i) for i in args.labels.split(","))
    num_data = args.num_train + args.num_validation + args.num_test

    dataset = PU(
        datadir=args.datadir,
        load=args.load,
        data_length=args.data_length,
        labels=label_set_list,
        window=args.window,
        normalization=args.normalization,
        backbone=args.backbone,
        number=num_data
    )

    return dataset
