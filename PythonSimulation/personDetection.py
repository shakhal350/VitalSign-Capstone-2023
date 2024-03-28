import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

from SVD_processing import SVD_Matrix


def apply_windowing(time_signal):
    # Hamming window
    window = np.hamming(len(time_signal))
    return time_signal * window


def compute_fft(time_signal, fRate):
    # Subtract the mean to center the signal
    time_signal_centered = time_signal - np.mean(time_signal)
    # Apply windowing
    time_signal_windowed = apply_windowing(time_signal_centered)

    # Compute FFT
    fft_signal = np.fft.fft(time_signal_windowed)
    fft_freq = np.fft.fftfreq(len(fft_signal), d=1 / fRate)
    # Keep only the positive frequencies
    half = len(fft_signal) // 2
    fft_signal = fft_signal[:half]
    fft_freq = fft_freq[:half]

    return fft_signal, fft_freq


def analyze_static(spectrogram_array, radar_parameters, threshold=0.005):
    # Initialize an array to store where significant variations occur
    static_array = np.zeros_like(spectrogram_array, dtype=bool)
    count_array = np.zeros(spectrogram_array.shape[1])

    # Iterate through each frequency bin
    for freq_bin in range(spectrogram_array.shape[1]):
        # Iterate over time frames, starting from the second frame
        for time_frame in range(1, spectrogram_array.shape[0]):
            # Calculate the relative change in magnitude
            magnitude_change = np.abs(spectrogram_array[time_frame, freq_bin] - spectrogram_array[time_frame - 1, freq_bin])
            relative_change = magnitude_change / spectrogram_array[time_frame - 1, freq_bin]

            # Check if the relative change exceeds the threshold
            if relative_change < threshold and spectrogram_array[time_frame, freq_bin] > 0.25:
                count_array[freq_bin] += 1
                static_array[time_frame, freq_bin] = True
    plt.imshow(static_array, aspect='auto', origin='lower')
    plt.xlabel('Frequency Bins')
    plt.ylabel('Time Frames')
    plt.title('Variation Detected')
    plt.show()
    # pick the top frequency bins with the with a high count
    freq_location_index = np.argsort(count_array)[::-1][:5]
    print(freq_location_index)
    return static_array, freq_location_index


def detect_person_by_svd(data, radar_parameters):
    # Apply SVD to data matrix
    noise_reduced_data = SVD_Matrix(data, radar_parameters, 1000)
    # Convert the reduced data to a spectrogram
    spectrogram_array = fft_spectrogram(noise_reduced_data, radar_parameters)
    # Logic to analyze variation in magnitude over 5 seconds
    static_array, freq_location = analyze_static(spectrogram_array, radar_parameters)

    if freq_location is not None:
        return freq_location  # Return the detected frequency location
    return False  # Indication that no static object was detected


# Updated fft_spectrogram function with range calculation
def fft_spectrogram(data, radar_parameters):
    # Extract radar parameters
    frameRate = radar_parameters["frameRate"]
    samplesPerFrame = radar_parameters["samplesPerFrame"]
    freqSlope = radar_parameters["freqSlope"]
    rangeMax = radar_parameters["rangeMax"]
    c = 3e8  # Speed of light (m/s)

    # Perform FFT on each frame of data
    spectrogram = []
    for i in range(0, len(data), samplesPerFrame):
        frame = data[i:i + samplesPerFrame]
        spectrum, _ = compute_fft(frame, frameRate)
        spectrogram.append(spectrum)
    spectrogram_array = np.abs(np.array(spectrogram))

    # Normalize the spectrogram
    spectrogram_array = spectrogram_array / np.max(spectrogram_array)

    # Calculate corresponding ranges
    times = len(data) / samplesPerFrame / frameRate
    # Calculate max beat frequency (f_max)
    f_max = (2 * rangeMax * freqSlope) / c
    # Frequency increment per bin (assuming linear mapping)
    delta_f = f_max / (samplesPerFrame // 2)  # only half the FFT samples represent positive frequencies
    # Calculate range for each FFT bin
    range_bins = np.array([(c * (i * delta_f)) / (2 * freqSlope) for i in range(samplesPerFrame // 2)])

    # Plot spectrogram with range axis
    plt.figure(figsize=(20, 8))
    plt.imshow(spectrogram_array, aspect='auto', origin='lower', extent=[range_bins[0], range_bins[-1], 0, times])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Range (m)')
    plt.ylabel('Time (s)')
    plt.title('Spectrogram')
    plt.show()

    return spectrogram_array
