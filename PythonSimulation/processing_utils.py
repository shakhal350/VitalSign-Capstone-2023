import numpy as np
from scipy.signal import find_peaks


def apply_windowing(time_signal):
    window = np.hamming(len(time_signal))
    return time_signal * window


def compute_fft(time_signal, fRate):
    time_signal_centered = time_signal - np.mean(time_signal)
    time_signal_windowed = apply_windowing(time_signal_centered)
    fft_signal = np.fft.fft(time_signal_windowed)
    fft_freq = np.fft.fftfreq(len(fft_signal), d=1 / fRate)[:len(fft_signal) // 2]
    return fft_signal[:len(fft_signal) // 2], fft_freq


def find_significant_peaks(fft_data, fft_freq, prominence=0.1, width=3, percentile=99):
    fft_data = np.abs(fft_data)
    threshold = np.percentile(fft_data, percentile)
    peaks, properties = find_peaks(fft_data, prominence=prominence, width=width, height=threshold)
    return peaks, properties


def select_best_peak(peaks, properties, fft_freq):
    if len(peaks) > 0:
        scores = properties['prominences'] * properties['widths'] * properties['peak_heights']
        best_peak_index = np.argmax(scores)
        return fft_freq[peaks[best_peak_index]]
    else:
        return 0


def determine_frequency_window(best_peak_freq, fft_freq, window_gap=0.1):
    start_freq = max(best_peak_freq - window_gap, 0)
    end_freq = best_peak_freq + window_gap
    start_index, end_index = np.searchsorted(fft_freq, [start_freq, end_freq])
    return start_freq, end_freq, start_index, end_index
