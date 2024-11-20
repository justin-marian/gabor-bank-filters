"""include/plots.py"""

import os
import scipy
import numpy as np
import matplotlib.pyplot as plt

from src.include.gabor_filters import gaussian_filter, gabor_filter, gabor_filter_bank

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../images"))
os.makedirs(BASE_DIR, exist_ok=True)
BASE_NAME = os.path.join(BASE_DIR, "")


def plot_gaussian_filter(size: int, sigma: float) -> None:
    """Plot the Gaussian filter response in the time domain, showing amplitude."""
    global BASE_NAME

    gaussian_response: np.ndarray = gaussian_filter(size, sigma)

    plt.plot(gaussian_response, linewidth=1)
    plt.title("Gaussian Filter Response", fontsize=14, fontweight='bold')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.yticks(np.arange(0, 0.022, 0.002))
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{BASE_NAME}Gaussian.png")
    plt.show()

def plot_gabor_time_domain(size: int, sigma: float, frequency: float) -> None:
    """Plot the Gabor filter response in the time domain for cosine and sine components."""
    global BASE_NAME

    cos_component, sin_component = gabor_filter(size, sigma, frequency)

    # Cosine component
    plt.plot(cos_component * 1e3, linewidth=1)
    plt.title("Gabor Filter - Cosine Component", fontsize=12, fontweight='bold')
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Amplitude (x10⁻³)", fontsize=10)
    plt.xticks(np.arange(0, size + 1, 200))
    plt.yticks(np.arange(-2, 2.0, 0.5))
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{BASE_NAME}Gabor Cosine.png")
    plt.show()

    # Sine component
    plt.plot(sin_component * 1e3, linewidth=1)
    plt.title("Gabor Filter - Sine Component", fontsize=12, fontweight='bold')
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Amplitude (x10⁻³)", fontsize=10)
    plt.xticks(np.arange(0, size + 1, 200))
    plt.yticks(np.arange(-2, 2.5, 0.5))
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{BASE_NAME}Gabor Sine.png")
    plt.show()

def plot_spectrum(A: float, B: float, M: int, fs: float, size: int) -> None:
    """Plot the spectrum of Gabor filter cosine and sine components."""
    global BASE_NAME

    freq_axis, gabor_filters = gabor_filter_bank(size, A, B, M, fs)
    freq_half = freq_axis[:size // 2]

    plt.figure(figsize=(12, 8))
    plt.title("Gabor Filters Spectrum", fontsize=14, fontweight='bold')

    colormap = plt.get_cmap('tab10')
    colors = [colormap(i % 10) for i in range(M)]

    for i in range(M):
        cos_component = gabor_filters[i][0]
        sin_component = gabor_filters[i][1]

        cos_spectrum = np.abs(scipy.fft.fft(cos_component)[: len(cos_component) // 2])
        sin_spectrum = np.abs(scipy.fft.fft(sin_component)[: len(sin_component) // 2])

        plt.plot(freq_half, cos_spectrum, linestyle='-', color=colors[i], label=f'Cosine {i + 1}')
        plt.plot(freq_half, sin_spectrum, linestyle='--', color=colors[i], label=f'Sine {i + 1}')

    plt.xlim(0, size // 2)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{BASE_NAME}Gabor Spectrum.png")
    plt.show()

def plot_mel_to_normal_mapping(mel_units: np.ndarray, hz_units: np.ndarray) -> None:
    """Plot the mapping between Mel and normal (Hz) scales."""
    global BASE_NAME

    scaling_factor: float = max(hz_units) / max(mel_units)
    fig, ax1 = plt.subplots(figsize=(8, 10))

    # Plot lines between Mel and Hz values
    for mel_val, hz_val in zip(mel_units, hz_units):
        ax1.plot([1, 2], [mel_val * scaling_factor, hz_val],
                 color='blue', linestyle='-', linewidth=2, marker='o', markersize=5)

    subset_indices = slice(3, 8)
    mel_subset = mel_units[subset_indices]
    hz_subset = hz_units[subset_indices]

    for i, (mel_val, hz_val) in enumerate(zip(mel_subset, hz_subset), start=4):
        ax1.plot([1, 2], [mel_val * scaling_factor, hz_val],
                 color='blue', linestyle='--', marker='*', markersize=10)
        ax1.text(0.7, mel_val * scaling_factor,
                 f"$c_{{{i}}} = {int(mel_val)}$", verticalalignment='center', fontsize=10)
        ax1.text(2.1, hz_val, f"{int(hz_val)} Hz", verticalalignment='center', fontsize=10)

    # y-axis for Mel scale
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Mel', 'Hz'])
    ax1.set_xlim(0.5, 2.5)
    ax1.set_title("Mel to Hz Mapping", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Mel Scale", fontsize=12)
    ax1.set_yticks(mel_units * scaling_factor)
    ax1.set_yticklabels([f"{int(mel)}" for mel in mel_units])
    ax1.grid(False)
    # y-axis for Hz scale
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(hz_units)
    ax2.set_yticklabels([f"{int(hz)} Hz" for hz in hz_units])
    ax2.set_ylabel("Hz Scale", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{BASE_NAME}Mel to Hz Mapping.png")
    plt.show()
