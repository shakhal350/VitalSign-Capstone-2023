import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from filter_processing import butter_bandpass_filter, apply_high_pass_filter
from peak_processing import find_significant_peaks, reconstruct_signal_from_peaks

accumulated_breathing_signal = []
accumulated_heart_signal = []


def setup_plots(plotnumber):
    if plotnumber == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
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
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 5))
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
    fps = radar_parameters["frameRate"]
    freqSlope = radar_parameters["freqSlope"]
    c = 3e8

    starting_sample = max(0, current_sample - sample_window_size)
    ending_sample = min(current_sample, len(data))

    # Data windowing
    time_data = data[starting_sample:ending_sample]

    # Perform FFT
    fft_freqs, fft_half_length, fft_magnitude = perform_FFT(c, fps, freqSlope, time_data)

    wrapped_phase = np.arctan2(np.imag(time_data), np.real(time_data))

    starting_time = starting_sample / sample_per_second  # Time of the first data point
    latest_time = ending_sample / sample_per_second  # Time of the latest data point

    # Calculate the time values for the x-axis
    time_values = np.linspace(starting_time, latest_time, num=len(time_data))

    line1.set_data(time_values, time_data)
    ax1.set_xlim(starting_time, latest_time)
    ax1.set_ylim(np.min(time_data), np.max(time_data))

    # Set the data for the plot
    line2.set_data(fft_freqs, fft_magnitude)
    ax2.set_xlim(0, fps / 2)
    ax2.set_ylim(0, np.max(fft_magnitude[1:fft_half_length]))

    return line1, line2


def perform_FFT(c, fps, freqSlope, time_data):
    # Perform FFT
    fft_complex = np.fft.fft(time_data)
    fft_normalized = fft_complex / np.max(np.abs(fft_complex))  # Normalizing by the max magnitude
    fft_half_length = len(fft_normalized) // 2
    # fft_half_length = len(fft_normalized)
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
    return fft_freqs, fft_half_length, fft_magnitude


def plot_filtered_fft(current_sample, data, radar_parameters, sample_window_size, line1, line2, line3, line4, ax1, ax2, ax3, ax4, label1, label2):
    fps = radar_parameters["frameRate"]
    sample_per_second = radar_parameters["samplesPerSecond"]
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

    starting_time = starting_sample / sample_per_second
    latest_time = ending_sample / sample_per_second
    ############################################################################################################
    ############################################ Breathing Rate ################################################
    ############################################################################################################
    if len(peak_index_breathing) > 0:
        peak_indices_heart = [idx for idx, _ in peak_index_breathing[:5]]
        print(f"Five Strongest peaks for Breathing = {freqs[peak_indices_heart[:5]]} Hz = ", freqs[peak_indices_heart[:5]] * 60, " breaths per minute")
        breathing_time = reconstruct_signal_from_peaks(breathing_fft, peak_indices_heart, freqs, len(time_data))
        # Append the new breathing time data to the accumulated data
        accumulated_breathing_signal.extend(breathing_time)
        # Ensure that time_values and accumulated_breathing_signal have the same shape
        time_values = np.linspace(0, latest_time, num=len(accumulated_breathing_signal))
        line3.set_data(time_values, accumulated_breathing_signal)
        ax3.set_xlim(0, latest_time)
        ax3.set_ylim(min(accumulated_breathing_signal), max(accumulated_breathing_signal))
    else:
        print("No significant breathing peaks found.")
    ############################################################################################################
    ############################################ Heart Rate ####################################################
    ############################################################################################################
    if len(peak_index_heart) > 0:
        peak_indices_breath = [idx for idx, _ in peak_index_heart[:5]]
        print(f"Five Strongest peak for Heart = {freqs[peak_indices_breath[:5]]} Hz = ", freqs[peak_indices_breath[:5]] * 60, " beats per minute")
        heart_time = reconstruct_signal_from_peaks(heart_fft, peak_indices_breath, freqs, len(time_data))
        accumulated_heart_signal.extend(heart_time)
        time_values = np.linspace(0, latest_time, num=len(accumulated_heart_signal))
        line4.set_data(time_values, accumulated_heart_signal)
        ax4.set_xlim(0, latest_time)
        ax4.set_ylim(min(accumulated_heart_signal), max(accumulated_heart_signal))
    else:
        print("No significant heart peaks found.")
    extra_gap = 0.3
    # Update the axes using the line data
    ax1.set_xlim(0, breathing_rate_range[1] + extra_gap)
    ax1.set_ylim(0, np.max(breathing_fft))
    ax2.set_xlim(heart_rate_range[0] - extra_gap, heart_rate_range[1] + extra_gap)
    ax2.set_ylim(0, np.max(heart_fft))

    heartRate = sum(freqs[peak_indices_heart[:5]] * 60) / len(peak_indices_heart)
    breathingRate = sum(freqs[peak_indices_breath[:5]] * 60) / len(peak_indices_breath)

    label1.config(text=f"{round(breathingRate, 2)} BPM")
    label2.config(text=f"{round(heartRate, 2)} BPM")

    return line1, line2, line3, line4


def create_animation(fig, fig2, ax1, ax2, ax3, ax4, ax5, ax6, line1, line2, line3, line4, line5, line6, label1, label2, data_Re, data_Im, radar_parameters, update_interval, timeWindowMultiplier=1):
    window_size = int(radar_parameters["samplesPerFrame"] * radar_parameters["frameRate"] * timeWindowMultiplier)
    print(f"Window Size: {window_size}")

    data_Re_filtered = apply_high_pass_filter(data_Re, 0.1, radar_parameters["frameRate"])
    data_Im_filtered = apply_high_pass_filter(data_Im, 0.1, radar_parameters["frameRate"])

    data = data_Re_filtered + 1j * data_Im_filtered
    frames = np.arange(window_size, len(data_Re), window_size)

    ani1 = FuncAnimation(fig,
                         lambda frame: plot_range_fft(frame, data, radar_parameters, window_size, line1, line2,
                                                      ax1, ax2), frames=frames, blit=False,
                         interval=update_interval * 1000, repeat=False)

    # ani1.save()

    ani2 = FuncAnimation(fig2, lambda frame: plot_filtered_fft(frame, data, radar_parameters, window_size, line3, line4, line5, line6,
                                                               ax3, ax4, ax5, ax6, label1, label2), frames=frames, blit=False,
                         interval=update_interval * 1000, repeat=False)
    ani2.save()