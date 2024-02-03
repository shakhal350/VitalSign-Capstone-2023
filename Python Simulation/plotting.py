import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from filter_processing import butter_bandpass_filter
from peak_processing import find_significant_peaks, ifft_from_peaks

phase_history = []


def setup_plots(plotnumber):
    if plotnumber == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        line1, = ax1.plot([], [], lw=0.5)
        line2, = ax2.plot([], [], lw=1)

        ax1.set_title('Magnitude of Signal Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Magnitude')

        ax2.set_title('FFT of Signal')
        ax2.set_xlabel('Freq (Hz)')
        ax2.set_ylabel('Magnitude')

        plt.tight_layout()
        return fig, ax1, ax2, line1, line2
    elif plotnumber == 2:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        line1, = ax1.plot([], [], lw=2)
        line2, = ax2.plot([], [], lw=2)
        line3, = ax3.plot([], [], lw=2)
        line4, = ax4.plot([], [], lw=2)

        ax1.set_title('Filtered Breathing Rate FFT')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')

        ax2.set_title('Filtered Heart Rate FFT')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')

        ax3.set_title('Time domain Breathing Rate')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Magnitude')

        ax4.set_title('Time domain Heart Rate')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Magnitude')

        plt.tight_layout()
        return fig, ax1, ax2, line1, line2, ax3, ax4, line3, line4


def plot_range_fft(current_sample, data, radar_parameters, sample_window_size, line1, line2, ax1, ax2):
    sample_per_second = radar_parameters["samplesPerSecond"]
    samplingRate = radar_parameters["sampleRate"]
    range_max = radar_parameters["rangeMax"]
    range_bins = radar_parameters["NumOfRangeBins"]
    range_res = radar_parameters["rangeResol"]
    fps = radar_parameters["frameRate"]
    Bandwidth = radar_parameters["bandwidth"]
    chirp_length = radar_parameters["chirpTime"]
    freqSlope = radar_parameters["freqSlope"]
    c = 3e8
    # print(
    #     f"sample_per_second: {sample_per_second},samplingRate: {samplingRate}, range_max: {range_max}, range_bins: {range_bins}, range_res: {range_res}, fps: {fps}, Bandwidth: {Bandwidth}, chirp_length: {chirp_length}")

    starting_sample = max(0, current_sample - sample_window_size)
    ending_sample = min(current_sample, len(data))

    # Data windowing
    time_data = data[starting_sample:ending_sample]

    # Perform FFT
    fft_complex = np.fft.fft(time_data)
    fft_normalized = fft_complex / np.max(np.abs(fft_complex))  # Normalizing by the max magnitude
    fft_half_length = len(fft_normalized) // 2
    fft_magnitude = np.abs(fft_normalized[:fft_half_length])  # Get magnitude spectrum for the first half
    fft_freqs = np.fft.fftfreq(len(time_data), 1 / fps)[:fft_half_length]
    # frequency_resolution = Sampling Rate / Number of Samples of FFT

    # Find the peak in the FFT
    peak_index = find_significant_peaks(fft_magnitude, fft_freqs, width=100)

    if len(peak_index) > 0:
        peak_index = peak_index[0][0]
        peak_Hz = fft_freqs[peak_index]
        # convert to meters
        distance = (peak_Hz * c) / (2 * freqSlope)
        print(f"Peak Frequency: {peak_Hz} Hz = {distance} meters")
    else:
        print("peak_index is empty")
        # handle the error appropriately

    starting_time = starting_sample / sample_per_second  # Time of the first data point
    latest_time = ending_sample / sample_per_second  # Time of the latest data point

    # Calculate the time values for the x-axis
    time_values = np.linspace(starting_time, latest_time, num=len(time_data))
    # print(f"Current Sample # {current_sample} at Time: {latest_time} seconds")

    line1.set_data(time_values, time_data)
    ax1.set_xlim(starting_time, latest_time)
    ax1.set_ylim(np.min(time_data), np.max(time_data))

    # Set the data for the plot
    line2.set_data(fft_freqs, fft_magnitude)
    ax2.set_xlim(0, fps / 2)
    ax2.set_ylim(0, np.max(fft_magnitude[1:fft_half_length]))

    return line1, line2


def plot_filtered_fft(current_sample, data, radar_parameters, sample_window_size, line1, line2, line3, line4, ax1, ax2, ax3, ax4):
    fps = radar_parameters["frameRate"]
    # create a function that takes in the current sample and plots the filtered fft using IIR filter to bandpass heart rate and breathing rate
    breathing_rate_range = [0.1, 0.8]  # Hz = 60*0.1 to 60*0.7 = 6 to 48 breaths per minute
    heart_rate_range = [0.8, 2]  # Hz = 60*0.8 to 60*2 = 48 to 120 beats per minute

    starting_sample = max(0, current_sample - sample_window_size)
    ending_sample = min(current_sample, len(data))

    # Data windowing
    time_data = data[starting_sample:ending_sample]


    # Apply bandpass filters
    breathing_filtered = butter_bandpass_filter(time_data, breathing_rate_range[0], breathing_rate_range[1], fps, order=5)
    heart_filtered = butter_bandpass_filter(time_data, heart_rate_range[0], heart_rate_range[1], fps, order=5)

    # Compute the FFT of magnitude of the filtered data
    breathing_fft = np.abs(np.fft.rfft(breathing_filtered))
    heart_fft = np.abs(np.fft.rfft(heart_filtered))
    freqs = np.fft.rfftfreq(len(time_data), 1.0 / fps)

    # Update the line objects for the plot
    line1.set_data(freqs, breathing_fft)
    line2.set_data(freqs, heart_fft)
    # only pick the beat frequency of the peaks found in the filtered fft for breathing and heart rate
    peak_index_breathing = find_significant_peaks(breathing_fft, freqs, width=100)
    peak_index_heart = find_significant_peaks(heart_fft, freqs, width=100)

    # print(f"Breathing Peaks: {peak_index_breathing}")
    # print(f"Heart Peaks: {peak_index_heart}")

    if len(peak_index_breathing) > 0:
        # Extract only the indices from the tuples
        peak_indices = [idx for idx, _ in peak_index_breathing[:5]]  # Take up to the first 5 indices

        peak_Hz_breathing = freqs[peak_indices]
        print(f"Breathing Rate: {peak_Hz_breathing} Hz")

        area_magnitudes = [area_magnitudes for _, area_magnitudes in peak_index_breathing[:5]]
        print(f"Magnitudes: {area_magnitudes}")

        # Assuming time_data is correctly defined elsewhere and ifft_from_peaks is correctly implemented
        breathing_time = ifft_from_peaks(peak_indices, area_magnitudes, len(breathing_filtered))

        line3.set_data(np.arange(len(breathing_time)), breathing_time)
        ax3.set_xlim(0, len(breathing_time))
        ax3.set_ylim(np.min(breathing_time), np.max(breathing_time))
    else:
        print("peak_index_breathing is empty")
        # handle the error appropriately

    if len(peak_index_heart) > 0:
        # Extract only the indices from the tuples
        peak_indices = [idx for idx, _ in peak_index_heart[:5]]  # Take up to the first 5 indices

        peak_Hz_breathing = freqs[peak_indices]  # Now this should work as expected
        print(f"Heart Rate: {peak_Hz_breathing} Hz")

        magnitudes = [magnitude for _, magnitude in peak_index_heart[:5]]
        print(f"Magnitudes: {magnitudes}")

        # Assuming time_data is correctly defined elsewhere and ifft_from_peaks is correctly implemented
        heart_time = ifft_from_peaks(peak_indices, magnitudes, len(heart_filtered))

        line3.set_data(np.arange(len(heart_time)), heart_time)
        ax3.set_xlim(0, len(heart_filtered))
        ax3.set_ylim(np.min(heart_filtered), np.max(heart_filtered))
    else:
        print("peak_index_breathing is empty")
        # handle the error appropriately

    # Update the axes using the line data
    ax1.set_xlim(breathing_rate_range[0], breathing_rate_range[1])
    ax1.set_ylim(0, np.max(breathing_fft))
    ax2.set_xlim(heart_rate_range[0], heart_rate_range[1])
    ax2.set_ylim(0, np.max(heart_fft))

    return line1, line2, line3, line4


def create_animation(data, radar_parameters, update_interval, timeWindowMultiplier=1):
    fig, ax1, ax2, line1, line2 = setup_plots(1)
    window_size = int(radar_parameters["samplesPerFrame"] * radar_parameters["frameRate"] * timeWindowMultiplier)
    print(f"Window Size: {window_size}")

    frames = np.arange(window_size, len(data), window_size)
    print(f"Frames: {frames}")

    ani1 = FuncAnimation(fig,
                         lambda frame: plot_range_fft(frame, data, radar_parameters, window_size, line1, line2,
                                                      ax1, ax2), frames=frames, blit=False,
                         interval=update_interval * 1000, repeat=False)

    plt.show()

    fig, ax1, ax2, line1, line2, ax3, ax4, line3, line4 = setup_plots(2)
    ani2 = FuncAnimation(fig,
                         lambda frame: plot_filtered_fft(frame, data, radar_parameters, window_size,
                                                         line1, line2, line3, line4, ax1, ax2, ax3, ax4), frames=frames,
                         blit=False,
                         interval=update_interval * 1000, repeat=False)
    plt.show()

    newFig, (newAx1, newAx2) = plt.subplots(2, 1)
    newAx1.plot(np.arange(len(phase_history)), phase_history)
    plt.show()
