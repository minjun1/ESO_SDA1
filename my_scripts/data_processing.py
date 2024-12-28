"""
data_processing.py

This script contains:
1) Functions to load file paths (from pickle)
2) A function to read and process H5 files from GCS
3) A function to clip and normalize data
4) A function to reshape (crop) data and combine event/noise sets
5) An example main() function that demonstrates the overall pipeline
"""

import pickle
import os
import numpy as np
import h5py
import gcsfs

def load_file_paths(pkl_filename):
    """
    Load file paths (list of strings) from a pickle file.

    Parameters
    ----------
    pkl_filename : str
        Path to the .pkl file containing the list of file paths

    Returns
    -------
    file_paths : list
        List of file paths
    """
    with open(pkl_filename, 'rb') as f:
        file_paths = pickle.load(f)
    print(f"File paths loaded from {pkl_filename}")
    return file_paths


def process_h5_files(
    file_paths,
    clip_percentile=85,
    normalization_range=(-2, 2)
):
    """
    Reads 'input' data from each H5 file in file_paths, clips extremes,
    and normalizes the data.

    Parameters
    ----------
    file_paths : list of str
        Paths to the H5 files
    clip_percentile : int
        Percentile used for clipping positive and negative extremes
    normalization_range : tuple (float, float)
        The target range after normalization

    Returns
    -------
    normalized_data_list : list of numpy.ndarray
        List of processed 2D arrays
    """
    fs = gcsfs.GCSFileSystem()
    normalized_data_list = []

    for file_path in file_paths:
        # Read the H5 file from GCS
        with fs.open(file_path, 'rb') as f:
            with h5py.File(f, 'r') as h5_file:
                input_data = h5_file['input'][:]  # 2D or 3D dataset

        input_np = np.array(input_data, dtype=np.float32)

        # Separate positive and negative values
        positive_values = input_np[input_np > 0]
        negative_values = input_np[input_np < 0]

        # Calculate percentiles
        if len(positive_values) > 0:
            percentile_positive = np.percentile(positive_values, clip_percentile)
        else:
            percentile_positive = 0

        if len(negative_values) > 0:
            percentile_negative = np.percentile(np.abs(negative_values), clip_percentile)
        else:
            percentile_negative = 0

        # Clip positive
        input_np_clipped = np.where(
            input_np > 0,
            np.clip(input_np, None, percentile_positive),
            input_np
        )
        # Clip negative in absolute terms
        input_np_clipped = np.where(
            input_np_clipped < 0,
            np.clip(input_np_clipped, -percentile_negative, None),
            input_np_clipped
        )

        # Normalize to the specified range
        def normalize(arr, min_val, max_val):
            arr_min = arr.min()
            arr_max = arr.max()
            # Avoid divide-by-zero if arr_max == arr_min
            if arr_max == arr_min:
                return np.zeros_like(arr) + (max_val + min_val) / 2.0
            norm_arr = (arr - arr_min) / (arr_max - arr_min)
            norm_arr = norm_arr * (max_val - min_val) + min_val
            return norm_arr

        input_np_normalized = normalize(input_np_clipped, normalization_range[0], normalization_range[1])
        normalized_data_list.append(input_np_normalized)

    return normalized_data_list


def ensure_uniformity_and_convert(
    data_list,
    crop_size1,
    crop_size2
):
    """
    Ensure all arrays in the list have the same shape (crop_size1, crop_size2),
    reshape them, and stack into a 4D array (N, crop_size1, crop_size2, 1).

    Parameters
    ----------
    data_list : list of numpy.ndarray
        List of data arrays, each presumably 2D.
    crop_size1 : int
    crop_size2 : int

    Returns
    -------
    data_array : numpy.ndarray
        4D array of shape (N, crop_size1, crop_size2, 1)
    """
    uniform_arrays = [
        img.reshape((crop_size1, crop_size2, 1))
        for img in data_list
        if img.shape == (crop_size1, crop_size2)
    ]
    if len(uniform_arrays) == 0:
        return np.array([])

    data_array = np.stack(uniform_arrays, axis=0)
    return data_array


def main_pipeline(
    train_event_pkl,
    train_noise_pkl,
    test_event_pkl,
    test_noise_pkl,
    crop_size1=288,
    crop_size2=695,
    clip_percentile=85,
    normalization_range=(-2, 2),
    out_dir='processed_data'
):
    """
    Full data processing pipeline:
    1. Load file paths from pickles.
    2. Process/clip/normalize each set (train/test, event/noise).
    3. Ensure uniform shape, combine data, save to .npy

    Parameters
    ----------
    train_event_pkl : str
    train_noise_pkl : str
    test_event_pkl : str
    test_noise_pkl : str
    crop_size1 : int
    crop_size2 : int
    clip_percentile : int
    normalization_range : tuple
    out_dir : str
        Folder for saving final .npy files

    Returns
    -------
    None. (Saves arrays to disk.)
    """

    # 1. Load file paths
    file_paths_train_event = load_file_paths(train_event_pkl)
    file_paths_train_noise = load_file_paths(train_noise_pkl)
    file_paths_test_event  = load_file_paths(test_event_pkl)
    file_paths_test_noise  = load_file_paths(test_noise_pkl)

    # 2. Process
    train_data_event = process_h5_files(file_paths_train_event, clip_percentile, normalization_range)
    train_data_noise = process_h5_files(file_paths_train_noise, clip_percentile, normalization_range)
    test_data_event  = process_h5_files(file_paths_test_event,  clip_percentile, normalization_range)
    test_data_noise  = process_h5_files(file_paths_test_noise,  clip_percentile, normalization_range)

    # 3. Ensure uniform shape
    train_event_uniform = ensure_uniformity_and_convert(train_data_event, crop_size1, crop_size2)
    train_noise_uniform = ensure_uniformity_and_convert(train_data_noise, crop_size1, crop_size2)
    test_event_uniform  = ensure_uniformity_and_convert(test_data_event,  crop_size1, crop_size2)
    test_noise_uniform  = ensure_uniformity_and_convert(test_data_noise,  crop_size1, crop_size2)

    # 4. Combine events/noise
    X_train = np.concatenate([train_event_uniform, train_noise_uniform], axis=0)
    y_train = np.concatenate([np.ones(len(train_event_uniform)), np.zeros(len(train_noise_uniform))], axis=0)

    X_test = np.concatenate([test_event_uniform, test_noise_uniform], axis=0)
    y_test = np.concatenate([np.ones(len(test_event_uniform)), np.zeros(len(test_noise_uniform))], axis=0)

    # 5. Save .npy
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)

    print(f"Saved processed arrays to folder: {out_dir}")


if __name__ == "__main__":
    # Example usage (optional):
    # Suppose you have .pkl files with the file paths in the current directory:
    train_event_pkl = 'train_event_files_thres40.pkl'
    train_noise_pkl = 'train_noise_files_thres40.pkl'
    test_event_pkl  = 'test_event_files_thres40.pkl'
    test_noise_pkl  = 'test_noise_files_thres40.pkl'

    # Run the main pipeline
    main_pipeline(
        train_event_pkl,
        train_noise_pkl,
        test_event_pkl,
        test_noise_pkl,
        crop_size1=288,
        crop_size2=695,
        clip_percentile=85,
        normalization_range=(-2, 2),
        out_dir='train_data_thres40'
    )
