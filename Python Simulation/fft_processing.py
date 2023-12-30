import numpy as np


def perform_fft(time_data, fps):
    fft_result = np.fft.fft(time_data)
    fft_freq = np.fft.fftfreq(len(fft_result), 1 / fps)
    fft_magnitude = np.abs(fft_result)
    return fft_freq, fft_magnitude


def apply_magnitude_cutoff(fft_magnitude, cutoff_threshold):
    fft_magnitude_cutoff = fft_magnitude.copy()
    fft_magnitude_cutoff[fft_magnitude < cutoff_threshold] = 0
    return fft_magnitude_cutoff


def perform_ifft(fft_magnitude):
    return np.fft.ifft(fft_magnitude)
