import numpy as np
from scipy.integrate import simps
from scipy.signal import find_peaks


def find_significant_peaks(fft_magnitude, fft_freqs, width):
    # print(f"fft_freqs[256] = 0.1 Hz: {fft_freqs[256]}")
    start_index = 256  # Starting index for slicing
    end_index = len(fft_magnitude) // 2  # End index for slicing, assuming you're working with a one-sided spectrum
    # Detect peaks
    peaks, _ = find_peaks(fft_magnitude[start_index:end_index], distance=100)
    # print(f"freq_resolution: {freq_resolution}")
    # Initialize list to store areas
    peak_areas = []

    # Calculate the area for each peak
    for peak in peaks:
        # print(f"Peak: {peak}")
        start_index = max(peak - width, 0)
        end_index = min(peak + width + 1, len(fft_magnitude))
        # print(f"start_index: {start_index}, end_index: {end_index}")
        # Calculate the area under the curve for this peak
        area = simps(fft_magnitude[start_index:end_index], fft_freqs[start_index:end_index])
        peak_areas.append((peak, area))

    # Sort peaks by area, descending
    significant_peaks = sorted(peak_areas, key=lambda x: x[1], reverse=True)

    return significant_peaks


def ifft_from_peaks(beat_frequencies, magnitudes, signal_length):
    """
    Reconstructs a time-domain signal from given frequencies and their magnitudes using IFFT.

    Parameters:
    - beat_frequencies: Array of beat frequencies (indices or actual frequencies)
    - magnitudes: Array of magnitudes corresponding to beat frequencies
    - signal_length: The length of the time-domain signal to reconstruct

    Returns:
    - Time-domain signal reconstructed from the specified frequencies and magnitudes
    """
    # Create a complex array for frequency spectrum, initialized with zeros
    freq_spectrum = np.zeros(signal_length, dtype=np.complex128)

    # Assuming beat_frequencies are indices, directly use them to set the magnitudes
    for freq, mag in zip(beat_frequencies, magnitudes):
        freq_spectrum[freq] = mag

        # Add the symmetric counterpart for negative frequencies
        if freq != 0:  # Avoid duplicating the DC component
            freq_spectrum[-freq] = mag

    # Perform the IFFT
    time_domain_signal = np.fft.ifft(freq_spectrum)

    # Return the real part of the reconstructed signal
    return np.real(time_domain_signal)
