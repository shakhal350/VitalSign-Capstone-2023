import numpy as np


def perform_fft(time_data, fps):
    """
        Computes the Fast Fourier Transform (FFT) of time-domain data.

        :param time_data: Time-domain data to be transformed
        :param fps: Frames per second of the time-domain data

        :return: fft_freq: Frequency values of the FFT
        :return: fft_magnitude: Magnitude values of the FFT

        """

    fft_result = np.fft.fft(time_data)
    fft_freq = np.fft.fftfreq(len(fft_result), 1 / fps)
    fft_magnitude = np.abs(fft_result)
    return fft_freq, fft_magnitude


def apply_magnitude_cutoff(fft_magnitude, cutoff_threshold):
    """
    :param fft_magnitude: Magnitude of the fft
    :param cutoff_threshold: Magnitude chosen for the cutoff TODO: Needs to be dynamic, currently is hardcoded

    :return fft_magnitude_cutoff: Magnitude of the fft with the cutoff applied
    """

    fft_magnitude_cutoff = fft_magnitude.copy()
    fft_magnitude_cutoff[fft_magnitude < cutoff_threshold] = 0
    return fft_magnitude_cutoff
