import numpy as np
from scipy.integrate import simps
from scipy.signal import find_peaks


def find_significant_peaks(fft_magnitude, fft_freqs, width):
    # Detect peaks
    peaks, _ = find_peaks(fft_magnitude, distance=10, prominence=0.1, width=1, height=0.1, threshold=0.1)
    # Initialize list to store areas
    peak_areas = []

    # Calculate the area for each peak
    for peak in peaks:
        start_index = max(peak - width, 0)
        end_index = min(peak + width + 1, len(fft_magnitude))
        # Calculate the area under the curve for this peak
        area = simps(fft_magnitude[start_index:end_index], fft_freqs[start_index:end_index])
        peak_areas.append((peak, area))

    # Sort peaks by area, descending
    significant_peaks = sorted(peak_areas, key=lambda x: x[1], reverse=True)

    return significant_peaks


def reconstruct_signal_from_peaks(fft_array, peak_indices, freqs, n_points):
    # Create a frequency domain representation with zeros
    fft_reconstructed = np.zeros(len(freqs), dtype=np.complex128)
    print(f"freqs[peak_indices]: {peak_indices}")
    # Set the magnitudes at the peak indices
    for idx in peak_indices:
        fft_reconstructed[idx] = fft_array[idx]

    fft_reconstructed[peak_indices] = fft_array[peak_indices]

    # Inverse FFT to get the time-domain signal
    reconstructed_signal = np.fft.ifft(fft_reconstructed[peak_indices], n_points)
    return reconstructed_signal
