import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

from filter_processing import butter_bandpass_filter
from peak_processing import find_significant_peaks

count = 0


def setup_plots(plotnumber):
    if plotnumber == 1:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10))
        line1, = ax1.plot([], [], lw=0.5)
        line2, = ax2.plot([], [], lw=1)

        ax1.set_title('Magnitude of Signal Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Magnitude')

        ax2.set_title('Range FFT of Signal')
        ax2.set_xlabel('Range (m)')
        ax2.set_ylabel('Magnitude')

        ax3.set_title('2D Range FFT of Signal')
        ax3.set_xlabel('Range (m)')
        ax3.set_ylabel('Time (s)')

        plt.tight_layout()
        return fig, ax1, ax2, ax3, line1, line2
    elif plotnumber == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        line1, = ax1.plot([], [], lw=2)
        line2, = ax2.plot([], [], lw=2)

        ax1.set_title('Filtered Breathing Rate FFT')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')

        ax2.set_title('Filtered Heart Rate FFT')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')

        plt.tight_layout()
        return fig, ax1, ax2, line1, line2


def plot_range_fft(current_sample, data, radar_parameters, sample_window_size, line1, line2, ax1, ax2, ax3, peak_text):
    samples_per_frame = radar_parameters["samplesPerFrame"]
    range_max = radar_parameters["rangeMax"]
    range_bin = radar_parameters["rangeBin"]
    fps = radar_parameters["frameRate"]
    starting_sample = max(0, current_sample - sample_window_size)
    ending_sample = min(current_sample, len(data))

    # Data windowing
    time_data = data[starting_sample:ending_sample]

    # Perform FFT
    fft_complex = np.fft.fft(time_data)
    fft_normalized = fft_complex / np.max(np.abs(fft_complex))  # Normalizing by the max magnitude
    fft_half_length = len(fft_normalized) // 2
    fft_magnitude = np.abs(fft_normalized[:fft_half_length])  # Get magnitude spectrum for the first half
    fft_phase = np.angle(fft_normalized[:fft_half_length])  # Get phase spectrum for the first half
    fft_freq = np.fft.fftfreq(len(time_data), 1 / fps)[:fft_half_length]

    # print(f"fft_phase: {fft_phase.shape}, fft_freq: {fft_freq.shape}, fft_magnitude: {fft_magnitude.shape}")

    # Ex: 512samples/frames * 20frames/sec = 10240 samples/sec
    sample_per_second = samples_per_frame * fps
    starting_time = starting_sample / sample_per_second  # Time of the first data point
    latest_time = ending_sample / sample_per_second  # Time of the latest data point

    # Calculate the time values for the x-axis
    time_values = np.linspace(starting_time, latest_time, num=len(time_data))
    print(f"Current Sample #{current_sample} at Time: {latest_time} seconds")

    # Find the significant peaks - adjust parameters as needed
    best_peak_indices = find_significant_peaks(fft_magnitude, min_distance=2, min_height=0.2, min_prominence=1)
    best_peak_freqs = fft_freq[best_peak_indices]
    # peak_text.set_text(f"Peak Frequency: {best_peak_freqs:.2f} Hz")

    line1.set_data(time_values, time_data)
    ax1.set_xlim(starting_time, latest_time)
    ax1.set_ylim(np.min(time_data), np.max(time_data))

    # Set the data for the plot
    line2.set_data(fft_freq, fft_magnitude)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, np.max(fft_magnitude[1:fft_half_length]))

    # Create original range_bins
    original_range_bins = np.linspace(0, range_max, range_bin)

    # Interpolate to match the length of fft_magnitude[:fft_half_length]
    interpolator = interp1d(np.linspace(0, 1, len(original_range_bins)), original_range_bins)
    interpolated_range_bins = interpolator(np.linspace(0, 1, fft_half_length))


    # For the 2D plot, we'll append the current fft_magnitude to the 'ax2' plot as a new row
    if 'fft_2d_data' not in ax3.__dict__:
        ax3.fft_2d_data = np.zeros((0, fft_half_length))
    # Append the current magnitude data as a new row in the 2D data array
    ax3.fft_2d_data = np.vstack((ax3.fft_2d_data, fft_magnitude[:fft_half_length]))

    # Update the 2D plot
    # Since ax3.fft_2d_data is now a 1D array, we need to reshape it to a 2D array with one row for imshow
    ax3.imshow(ax3.fft_2d_data.T, aspect='auto', extent=[0, latest_time, 0, range_max], origin='lower', cmap='jet', vmin=0, vmax=0.2)
    ax3.set_ylim(0, 2)
    # Set the labels and title
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Range (m)')
    ax3.set_title('Current Range FFT Magnitude Over Time')

    return line1, line2, peak_text


# def plot_isolated_bin(current_sample, data, samples_per_frame, range_max, range_bin, fps, sample_window_size, line1, line2, ax1, ax2, peak_text):


def plot_filtered_fft(current_sample, data, radar_parameters, sample_window_size, line1, line2, ax1, ax2, peak_text):
    samples_per_frame = radar_parameters["samplesPerFrame"]
    fps = radar_parameters["frameRate"]
    # create a function that takes in the current sample and plots the filtered fft using IIR filter to bandpass heart rate and breathing rate
    breathing_rate_range = [0.1, 0.5]  # Hz
    heart_rate_range = [0.8, 2]  # Hz

    starting_sample = max(0, current_sample - sample_window_size)
    ending_sample = current_sample

    # TODO: Write code to filter the data using IIR filter
    # Extract the segment of data we're going to filter and FFT
    data_segment = data[starting_sample:ending_sample]

    # Apply bandpass filters
    breathing_filtered = butter_bandpass_filter(data_segment, breathing_rate_range[0], breathing_rate_range[1], fps, order=5)
    heart_filtered = butter_bandpass_filter(data_segment, heart_rate_range[0], heart_rate_range[1], fps, order=5)

    # Compute the FFT of magnitude of the filtered data
    breathing_fft = np.abs(np.fft.rfft(breathing_filtered))
    heart_fft = np.abs(np.fft.rfft(heart_filtered))
    freqs = np.fft.rfftfreq(len(data_segment), 1.0 / fps)

    # Update the line objects for the plot
    line1.set_data(freqs, breathing_fft)
    line2.set_data(freqs, heart_fft)

    # Update the axes using the line data
    ax1.set_xlim(breathing_rate_range[0], breathing_rate_range[1])
    ax1.set_ylim(0, np.max(breathing_fft))
    ax2.set_xlim(heart_rate_range[0], heart_rate_range[1])
    ax2.set_ylim(0, np.max(heart_fft))

    return line1, line2, peak_text


def create_animation(data, radar_parameters, window_size, update_interval):
    fig, ax1, ax2, ax3, line1, line2 = setup_plots(1)

    # Create a text artist for peak annotations
    peak_text = ax2.text(2, 2, '', transform=ax2.transAxes, horizontalalignment='right', verticalalignment='top', color='red')
    frames = np.arange(window_size, len(data), window_size)

    ani1 = FuncAnimation(fig,
                         lambda frame: plot_range_fft(frame, data, radar_parameters, window_size, line1, line2,
                                                      ax1, ax2, ax3, peak_text), frames=frames, blit=False,
                         interval=update_interval * 1000, repeat=False)

    plt.show()

    fig, ax1, ax2, line1, line2 = setup_plots(2)
    ani2 = FuncAnimation(fig,
                         lambda frame: plot_filtered_fft(frame, data, radar_parameters, window_size,
                                                         line1, line2, ax1, ax2, peak_text), frames=frames,
                         blit=True,
                         interval=update_interval * 1000, repeat=False)
    plt.show()
