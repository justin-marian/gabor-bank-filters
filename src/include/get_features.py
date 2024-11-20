"""include/get_features.py"""

import numpy as np

from src.include.gabor_filters import from_hz_to_mel, from_mel_to_hz, gabor_filter_bank
from src.include.plots import plot_gaussian_filter, plot_spectrum, plot_gabor_time_domain, plot_mel_to_normal_mapping

PLOT_ONLY_ONCE: bool = False


def get_features(audio_train: np.ndarray, fs: float) -> np.ndarray:
    """
    Extract frequency-based features from audio signals.
    Computes feature vectors for each file in the training and testing datasets.
        It receives all the audio signals from a dataset
        in a matrix of size [Dataset_size (D) x Number_of_samples (N)]
        and returns all features in a matrix of size [D x (2 * M)].
    """

    global PLOT_ONLY_ONCE

    # Gaussian filter parameters
    size_gaussian: int = 101
    sigma_gaussian: float = 20.0

    # Gabor filter parameters
    size_gabor: int = 1102
    sigma_i: float = 187.21221
    central_frequency: float = 0.00267

    # Window size, number of filters, and step size for segmentation
    window_size: int = 1102
    num_filters: int = 12
    step_size: int = int(12 * 10 ** (-3) * fs)  # Step size is 12 ms

    # Frequency range in the Mel scale
    mel_start: float = from_hz_to_mel(0)
    mel_end: float = from_hz_to_mel(fs / 2)

    # Conversion for Mel-to-Hertz mapping
    mel_units: np.ndarray = np.arange(1000, 3100, 200)
    hz_units: np.ndarray = from_mel_to_hz(mel_units)

    # Plot and save images only once (for the test set)
    if PLOT_ONLY_ONCE:
        plot_gaussian_filter(size_gaussian, sigma_gaussian)
        plot_gabor_time_domain(size_gabor, sigma_i, central_frequency)
        plot_spectrum(mel_start, mel_end, num_filters, fs, size_gabor)
        plot_mel_to_normal_mapping(mel_units, hz_units)
        PLOT_ONLY_ONCE = False  # Avoid repeated saving

    # Generate Gabor filter bank
    freq_axis, gabor_filters = gabor_filter_bank(window_size, mel_start, mel_end, num_filters, fs)
    gabor_filters_matrix: np.ndarray = np.vstack(gabor_filters)  # (M, K)

    # Compute features for each audio signal
    features_list: list[np.ndarray] = []
    for audio_signal in audio_train:
        # Number of windows (segments) in the audio signal
        num_windows: int = (len(audio_signal) - window_size) // step_size + 1

        # Create windows of size K by segmenting the signal
        windowed_data = np.array([
            audio_signal[idx * step_size: idx * step_size + window_size]
            for idx in range(num_windows)
        ])  # (F, K)

        # Compute convolution with Gabor filters (dot product with reversed filters)
        filtered_windows: np.ndarray = np.abs(windowed_data @ gabor_filters_matrix[:, ::-1].T)  # (F, M)

        # Compute mean and standard deviation for each filter (column-wise)
        mean_features: np.ndarray = np.mean(filtered_windows, axis=0)   # (1, M)
        std_features: np.ndarray = np.std(filtered_windows, axis=0)     # (1, M)

        # Concatenate mean and standard deviation into a single feature vector
        feature_vector: np.ndarray = np.hstack((mean_features, std_features))  # (1, 2M)
        features_list.append(feature_vector)

    PLOT_ONLY_ONCE = True
    return np.array(features_list)  # (D, 2M)
