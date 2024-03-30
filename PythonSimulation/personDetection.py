import numpy as np
import matplotlib.pyplot as plt

from SVD_processing import SVD_Matrix


def apply_windowing(time_signal):
    # Hamming window
    window = np.hamming(len(time_signal))
    return time_signal * window


def range_to_frequency(static_location_m, radar_parameters):
    c = 3e8  # Speed of light (m/s)
    rangeMax = radar_parameters["rangeMax"]
    freqSlope = radar_parameters["freqSlope"]
    samplesPerFrame = radar_parameters["samplesPerFrame"]

    # Calculate the maximum frequency
    f_max = (2 * rangeMax * freqSlope) / c
    # Calculate the frequency resolution
    delta_f = f_max / (samplesPerFrame // 2)
    # Convert range to frequency
    f = (2 * static_location_m * freqSlope) / c

    return f


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


def analyze_static_variance(spectrogram_array, time_bins, range_bins, radar_parameters, threshold_V=0.5, threshold_S=0.005, window_size=10):
    # Initialize arrays to store where significant variations occur
    static_array = np.zeros_like(spectrogram_array, dtype=bool)
    variance_array = np.zeros_like(spectrogram_array, dtype=bool)
    count_array_s = np.zeros(spectrogram_array.shape[1])
    count_array_v = np.zeros(spectrogram_array.shape[1])

    # Iterate through each frequency bin
    for freq_bin in range(spectrogram_array.shape[1]):
        # Iterate over time frames, considering the window size
        for time_frame in range(window_size - 1, spectrogram_array.shape[0]):
            # Calculate the average magnitude in the current window
            current_window_average = np.mean(spectrogram_array[time_frame - window_size + 1:time_frame + 1, freq_bin])
            previous_value = spectrogram_array[time_frame - window_size, freq_bin] if time_frame >= window_size else 0
            magnitude_change = np.abs(current_window_average - previous_value)
            relative_change = magnitude_change / previous_value if previous_value != 0 else 0

            # Check if the relative change exceeds the thresholds
            if relative_change > threshold_V and current_window_average < 0.15:
                count_array_v[freq_bin] += 1
                variance_array[time_frame, freq_bin] = True
            if relative_change < threshold_S and current_window_average > 0.25:
                count_array_s[freq_bin] += 1
                static_array[time_frame, freq_bin] = True

    # Plot results
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    # For static_array visualization
    ax[0].pcolormesh(range_bins, time_bins, variance_array, cmap='hot', shading='flat')
    ax[1].pcolormesh(range_bins, time_bins, static_array, shading='flat')

    ax[0].set_title('Variance Detection')
    ax[0].set_xlabel('Frequency Bin')
    ax[0].set_ylabel('Time Frame')

    ax[1].set_title('Static Object Detection')
    ax[1].set_xlabel('Frequency Bin')
    ax[1].set_ylabel('Time Frame')

    plt.tight_layout()
    plt.show()

    # Check through the count array for evidence of a static object
    static_location = None
    for i in range(len(count_array_s)):
        if count_array_s[i] > 0.1 * spectrogram_array.shape[0]:
            static_location = i
            break

    print("static_location: ", static_location)
    return static_location


def detect_person_by_svd(data, radar_parameters, svd_components):
    # Apply SVD to data matrix
    noise_reduced_data_re = SVD_Matrix(np.real(data), radar_parameters, svd_components)
    noise_reduced_data_im = SVD_Matrix(np.imag(data), radar_parameters, svd_components)
    noise_reduced_data = noise_reduced_data_re + 1j * noise_reduced_data_im
    # Convert the reduced data to a spectrogram
    spectrogram_array, time_bins, range_bins, fft_freq = fft_spectrogram(noise_reduced_data, radar_parameters)
    # Logic to analyze variation in magnitude over 5 seconds
    static_location = analyze_static_variance(spectrogram_array, time_bins, range_bins, radar_parameters)
    if static_location is None:
        return False
    else:
        freq_location_index = fft_freq[static_location]
        print("freq_location_index: ", freq_location_index)
        return freq_location_index




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
        spectrum, fft_freq = compute_fft(frame, frameRate)
        spectrogram.append(spectrum)
    spectrogram_array = np.abs(np.array(spectrogram))

    # Normalize the spectrogram
    spectrogram_array = spectrogram_array / np.max(spectrogram_array)

    # Calculate corresponding ranges and times correctly
    times = int(len(data) / (frameRate * samplesPerFrame))
    f_max = (2 * rangeMax * freqSlope) / c
    delta_f = f_max / (samplesPerFrame // 2)
    range_bins = np.array([(c * (i * delta_f)) / (2 * freqSlope) for i in range(samplesPerFrame // 2 + 1)])
    time_bins = np.linspace(0, times, spectrogram_array.shape[0] + 1)

    # Adjust range_bins if necessary to match the cut to 5m
    cut_index = int(5 * len(range_bins) / rangeMax)
    range_bins = range_bins[:cut_index + 1]
    spectrogram_array = spectrogram_array[:, :cut_index]

    # Plot spectrogram with range axis
    plt.figure(figsize=(20, 8))
    plt.pcolormesh(range_bins, time_bins, spectrogram_array, shading='auto')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Range (m)')
    plt.ylabel('Time (s)')
    plt.title('Spectrogram')
    plt.show()

    return spectrogram_array, time_bins, range_bins, fft_freq
