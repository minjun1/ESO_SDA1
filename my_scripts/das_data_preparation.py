"""
das_data_preparation.py

Contains functions to load and prepare DAS (Distributed Acoustic Sensing) data
for training and testing. This script includes:

1) Loading earthquake and noise catalogs from CSV files
2) Removing missing data
3) Filtering events based on amplitude threshold
4) Generating file paths for event/noise data
5) Splitting data into train/test sets
6) Saving/Loading file paths with pickle
"""

import os
import random
import pickle

import numpy as np
import pandas as pd
import gcsfs

random.seed(42)

def load_catalog_data(event_csv_path, noise_csv_path):
    """
    Load the event and noise catalogs from CSV files.

    Parameters:
    -----------
    event_csv_path : str
        Path to the earthquake catalog CSV file.
    noise_csv_path : str
        Path to the noise catalog CSV file.

    Returns:
    --------
    df_events : pandas.DataFrame
    df_noise : pandas.DataFrame
    """
    df_events = pd.read_csv(event_csv_path)
    df_noise = pd.read_csv(noise_csv_path)
    return df_events, df_noise


def remove_missing_data(df):
    """
    Remove rows from a DataFrame that are missing DAS or seismometer data.

    Parameters:
    -----------
    df : pandas.DataFrame

    Returns:
    --------
    df : pandas.DataFrame
        Filtered DataFrame.
    """
    df = df[df.has_das_data == True]
    df = df[df.has_seismometer_data == True]
    return df


def get_event_indices(df, threshold):
    """
    Filter the DataFrame to get rows where local_amplitude >= threshold
    and return their indices and amplitudes in a shuffled order.

    Parameters:
    -----------
    df : pandas.DataFrame
    threshold : float
        The amplitude threshold.

    Returns:
    --------
    shuffled_indices : list
        List of row indices passing the threshold.
    shuffled_amplitudes : list
        Corresponding local amplitudes in the same order.
    """
    filtered_df = df[df.local_amplitude >= threshold]
    
    # Get the indices and local_amplitude values
    indices = list(filtered_df.index)
    amplitudes = list(filtered_df.local_amplitude)
    
    # Shuffle the indices/amplitudes
    combined = list(zip(indices, amplitudes))
    random.shuffle(combined)
    
    # Separate them back
    shuffled_indices, shuffled_amplitudes = zip(*combined) if combined else ([], [])
    return list(shuffled_indices), list(shuffled_amplitudes)


def _get_filenames(indices, prefix, data_dir, fs=None, batch=1000):
    """
    Construct the HDF5 filenames corresponding to the event/noise data.

    Parameters:
    -----------
    indices : list
        List of row indices for events or noise.
    prefix : str
        'event' or 'noise' prefix for filenames.
    data_dir : str
        GCS or local directory containing the data subfolders.
    fs : gcsfs.GCSFileSystem or None
        File system object for checking GCS existence (optional).
    batch : int
        Subdirectory size grouping (e.g. 1000).

    Returns:
    --------
    filenames : list
        Valid file paths that exist in the filesystem.
    """
    if fs is None:
        fs = gcsfs.GCSFileSystem()

    filenames = []
    for i in indices:
        filename = f"{prefix}_{i:05d}_1.h5"
        subdir = f"{(i // batch) * batch:05d}"
        full_path = os.path.join(data_dir, prefix, subdir, filename).replace('\\', '/')
        
        if not fs.exists(full_path):
            print(f'File does not exist: {full_path}')
            continue
        
        filenames.append(full_path)

    return filenames


def save_file_paths(file_paths, filename):
    """
    Save file paths (list of strings) to a pickle file.

    Parameters:
    -----------
    file_paths : list
    filename : str
    """
    with open(filename, 'wb') as f:
        pickle.dump(file_paths, f)
    print(f"File paths saved to {filename}")


def load_file_paths(filename):
    """
    Load file paths (list of strings) from a pickle file.

    Parameters:
    -----------
    filename : str

    Returns:
    --------
    file_paths : list
    """
    with open(filename, 'rb') as f:
        file_paths = pickle.load(f)
    print(f"File paths loaded from {filename}")
    return file_paths


def prepare_das_data(
    event_csv_path,
    noise_csv_path,
    data_dir,
    amplitude_threshold=40,
    noise_ratio=10,
    train_ratio=0.8
):
    """
    Complete pipeline to load data, remove missing rows, filter events by amplitude,
    and produce train/test file path lists.

    Parameters:
    -----------
    event_csv_path : str
        Path to the earthquake catalog CSV.
    noise_csv_path : str
        Path to the noise catalog CSV.
    data_dir : str
        Path (GCS or local) to where the HDF5 data subfolders reside.
    amplitude_threshold : float
        Minimum local amplitude threshold for an event.
    noise_ratio : int
        Ratio of noise windows to event windows.
    train_ratio : float
        Fraction of data to be used for training (remainder is for testing).

    Returns:
    --------
    train_event_files, test_event_files,
    train_event_amplitudes, test_event_amplitudes,
    train_noise_files, test_noise_files
    """
    # 1. Load data
    df_events, df_noise = load_catalog_data(event_csv_path, noise_csv_path)

    # 2. Remove missing data
    df_events = remove_missing_data(df_events)
    df_noise = remove_missing_data(df_noise)

    # 3. Get event indices passing amplitude threshold
    event_indices, event_amplitudes = get_event_indices(df_events, amplitude_threshold)
    n_event = len(event_indices)

    # 4. Pick the appropriate noise slices
    noise_indices = df_noise.index[: n_event * noise_ratio]

    # 5. Split train/test for events
    n_event_train = int(train_ratio * n_event)
    n_event_test  = n_event - n_event_train

    train_event_indices = event_indices[:n_event_train]
    test_event_indices  = event_indices[n_event_train:]

    train_event_amplitudes = event_amplitudes[:n_event_train]
    test_event_amplitudes  = event_amplitudes[n_event_train:]

    # 6. Split train/test for noise
    n_noise_train = noise_ratio * n_event_train
    n_noise_test  = noise_ratio * n_event_test

    train_noise_indices = noise_indices[:n_noise_train]
    test_noise_indices  = noise_indices[n_noise_train : n_noise_train + n_noise_test]

    # 7. Get filenames
    fs = gcsfs.GCSFileSystem()
    train_event_files = _get_filenames(train_event_indices, "event", data_dir, fs)
    test_event_files  = _get_filenames(test_event_indices,  "event", data_dir, fs)

    train_noise_files = _get_filenames(train_noise_indices, "noise", data_dir, fs)
    test_noise_files  = _get_filenames(test_noise_indices,  "noise", data_dir, fs)

    # 8. Shuffle event files/amplitudes together
    train_event_combined = list(zip(train_event_files, train_event_amplitudes))
    test_event_combined  = list(zip(test_event_files,  test_event_amplitudes))

    random.shuffle(train_event_combined)
    random.shuffle(test_event_combined)

    if train_event_combined:
        train_event_files, train_event_amplitudes = zip(*train_event_combined)
    else:
        train_event_files, train_event_amplitudes = [], []
    
    if test_event_combined:
        test_event_files, test_event_amplitudes = zip(*test_event_combined)
    else:
        test_event_files, test_event_amplitudes = [], []

    # 9. Shuffle noise files
    train_noise_files = list(train_noise_files)
    test_noise_files  = list(test_noise_files)

    random.shuffle(train_noise_files)
    random.shuffle(test_noise_files)

    return (
        list(train_event_files),
        list(test_event_files),
        list(train_event_amplitudes),
        list(test_event_amplitudes),
        train_noise_files,
        test_noise_files
    )

if __name__ == "__main__":
    # Example usage (optional):  
    # Just a demonstration if you run `python das_data_preparation.py` directly.
    
    event_csv = "catalog/earthquake_catalog.csv"
    noise_csv = "catalog/noise_catalog.csv"
    gcs_data_dir = "gs://sep-data-backup/fantine/earthquake-detection-ml/processed_data/das/"
    
    (
        train_event_files,
        test_event_files,
        train_event_amplitudes,
        test_event_amplitudes,
        train_noise_files,
        test_noise_files
    ) = prepare_das_data(event_csv, noise_csv, gcs_data_dir)
    
    print("Number of training event files:", len(train_event_files))
    print("Number of testing event files:", len(test_event_files))
    print("Number of training noise files:", len(train_noise_files))
    print("Number of testing noise files:", len(test_noise_files))
