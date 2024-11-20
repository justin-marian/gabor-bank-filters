"""include/gabor_filters.py"""

import numpy as np
from typing import List, Tuple


def gabor_filter(size: int, sigmai: float, fi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a Gabor filter with both cosine and sine modulations."""
    gaussian: np.ndarray = gaussian_filter(size, sigmai)
    cos_h: np.ndarray = np.array([gaussian[i] * np.cos(2 * np.pi * fi * i) for i in range(size)])
    sin_h: np.ndarray = np.array([gaussian[i] * np.sin(2 * np.pi * fi * i) for i in range(size)])
    return cos_h, sin_h

def gabor_filter_bank(size: int, A: float, B: float, M: int, fs: float) \
        -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Generates a set of Gabor filters over a range of frequencies."""
    freq_axis: np.ndarray = np.linspace(0, size, size)
    mel_freq: np.ndarray = np.linspace(A, B, M + 1)
    hz_freq: List[float] = [from_mel_to_hz(f) for f in mel_freq]

    # Band segment lengths (l) and center frequencies (c) for each filter
    l: List[float] = [hz_freq[i] - hz_freq[i - 1] for i in range(1, M + 1)]
    c: List[float] = [(hz_freq[i] + hz_freq[i + 1]) / 2 for i in range(0, M)]

    # Generating Gabor filters for each segment defined by band lengths and center frequencies
    gabor_filters: List[Tuple[np.ndarray, np.ndarray]] = [
        gabor_filter(size, fs / l[i], c[i] / fs) for i in range(0, M)
    ]

    return freq_axis, gabor_filters

def gaussian_filter(size: int, sigma: float) -> np.ndarray:
    """Generates a Gaussian filter based on size and standard deviation."""
    mu: float = size / 2
    gaussian: np.ndarray = np.array([
        (1 / (sigma * np.sqrt(2 * np.pi))) *            # denominator
        np.exp(-((n - mu) ** 2) / (2 * sigma ** 2))     # numerator
        for n in range(size)
    ])
    return gaussian

def from_mel_to_hz(f_mel):
    """Converts a frequency from the Mel scale to Hertz."""
    return 700 * (np.exp(f_mel / 1127) - 1)

def from_hz_to_mel(f_hz):
    """Converts a frequency from Hertz to the Mel scale."""
    return 1127 * np.log(1 + f_hz / 700)
