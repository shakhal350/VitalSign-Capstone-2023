import numpy as np



def perform_fft(time_data, fps):
    """
        Computes the Fast Fourier Transform (FFT) of time-domain data.

        :param time_samples: Data points in the time domain
        :param sampling_rate: Frequency at which the data points were sampled

        :return: A tuple containing the frequency array, the magnitude of the FFT,
                 suitable for further analysis or plotting
        """

    fft_result = np.fft.fft(time_data)
    fft_freq = np.fft.fftfreq(len(fft_result), 1 / fps)
    fft_magnitude = np.abs(fft_result)
    return fft_freq, fft_magnitude


def apply_magnitude_cutoff(fft_magnitude, cutoff_threshold):
    fft_magnitude_cutoff = fft_magnitude.copy()
    fft_magnitude_cutoff[fft_magnitude < cutoff_threshold] = 0
    return fft_magnitude_cutoff
