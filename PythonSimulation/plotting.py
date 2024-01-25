import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

from fft_processing import perform_fft

# Peak data to global lists
global peak_frequencies, peak_magnitudes
peak_frequencies = []
peak_magnitudes = []


def setup_plots(plotnumber):
    if plotnumber == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))
        line1, = ax1.plot([], [], lw=2)
        line2, = ax2.plot([], [], lw=2)

        ax1.set_title('Magnitude of Signal Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Magnitude')

        ax2.set_title('FFT of Signal')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')

        plt.tight_layout()
        return fig, ax1, ax2, line1, line2
    elif plotnumber == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3))
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


def plot_time_fft(current_sample, data, samples_per_frame, fps, sample_window_size, line1, line2, ax1,
                  ax2,
                  peak_text):
    """
    Update the plot with the latest data.

    Parameters:
    current_sample (int): The index of the current sample.
    data (pandas.DataFrame): The data to be plotted.
    samples_per_frame (int): The number of samples per frame.
    fps (int): The frames per second.
    sample_window_size (int): The size of the sample window.
    line1 (matplotlib.lines.Line2D): The line object for the time plot.
    line2 (matplotlib.lines.Line2D): The line object for the FFT plot.
    ax1 (matplotlib.axes.Axes): The axes object for the time plot.
    ax2 (matplotlib.axes.Axes): The axes object for the FFT plot.
    peak_text (matplotlib.text.Text): The text object for the peak frequencies.

    Returns:
    line1 (matplotlib.lines.Line2D): The updated line object for the time plot.
    line2 (matplotlib.lines.Line2D): The updated line object for the FFT plot.
    peak_text (matplotlib.text.Text): The updated text object for the peak frequencies.
    """
    starting_sample = abs(current_sample - sample_window_size)
    ending_sample = current_sample

    print(f"len(data): {len(data)}, current_sample: {current_sample}, sample_window_size: {sample_window_size}")

    # Ex: 512samples/frames * 20frames/sec = 10240 samples/sec
    sample_per_second = samples_per_frame * fps

    latest_time = ending_sample / sample_per_second  # Time of the latest data point
    time_data = (data[starting_sample:ending_sample])
    if current_sample + sample_window_size > len(data):
        # Adjust ending_sample for the last frame to include all remaining data
        ending_sample = min(current_sample + sample_window_size, len(data))

        # Calculate the number of samples in the current window
        num_samples_in_window = ending_sample - starting_sample

        # If the remaining data is smaller than the sample window size, pad with zeros
        if num_samples_in_window < sample_window_size:
            # Calculate the number of samples to pad
            num_samples_to_pad = sample_window_size - num_samples_in_window

            # Pad the time_data with zeros
            time_data = np.concatenate((time_data, np.zeros(num_samples_to_pad)))

    # Calculate the time values for the x-axis
    time_values = np.linspace(starting_sample / sample_per_second, ending_sample / sample_per_second, num=len(time_data))
    print(f"Current Sample #{current_sample} at Time: {latest_time} seconds")

    # Set the data for the line plot with time_values as x-axis and time_data as y-axis
    line1.set_data(time_values, time_data)
    ax1.set_xlim(starting_sample / sample_per_second, ending_sample / sample_per_second)
    ax1.set_ylim(np.min(time_data), np.max(time_data))

    # Call FFT processing functions
    fft_freq, fft_magnitude = perform_fft(time_data, fps)
    line2.set_data(fft_freq[:len(fft_magnitude) // 2], fft_magnitude[:len(fft_magnitude) // 2])
    ax2.set_xlim(0, max(fft_freq))
    ax2.set_ylim(0, np.max(fft_magnitude[np.argmax(fft_freq):]))

    peaks, _ = find_peaks(fft_magnitude, height=0, distance=2)

    # min_freq_index = np.argmax(fft_freq >= min_freq_cutoff)
    #
    # # Pair peak frequencies with their magnitudes and sort by magnitude (descending)
    # peak_freqs_magnitudes = [(fft_freq[peak + min_freq_index],
    #                           fft_magnitude[peak]) for peak in peaks]
    # peak_freqs_magnitudes.sort(key=lambda x: x[1], reverse=True)

    # Convert frequencies to BPM and create annotation text
    # peak_freq_bpm_text = '\n'.join(
    #     f"{freq:.2f} Hz = {freq * 60:.2f} BPM" for freq, mag in peak_freqs_magnitudes)
    # peak_text.set_text(peak_freq_bpm_text)
    #
    # for freq, mag in peak_freqs_magnitudes:
    #     peak_frequencies.append(freq)
    #     peak_magnitudes.append(mag)

    return line1, line2, peak_text


def plot_filtered_fft(current_sample, data, samples_per_frame, fps, sample_window_size, line1, line2, ax1, ax2, peak_text):
    # create a function that takes in the current sample and plots the filtered fft using IIR filter to bandpass heart rate and breathing rate
    breathing_rate_range = [0.1, 0.5]  # Hz
    heart_rate_range = [0.8, 2]  # Hz

    starting_sample = max(0, current_sample - sample_window_size)
    ending_sample = current_sample

    # TODO: Write code to filter the data using IIR filter

    return line1, line2, peak_text


# def plot_histogram(peak_frequencies, peak_magnitudes):
#     # Convert frequencies to heart rates in BPM
#     heart_rates = [freq * 60 for freq in peak_frequencies]
#     if len(heart_rates) == 0:
#         print("No peaks found")
#         return
#     # Create a histogram of heart rates weighted by their magnitudes
#     plt.figure()
#     plt.hist(heart_rates, weights=peak_magnitudes, bins=range(
#         int(min(heart_rates)), int(max(heart_rates)) + 1, 1))
#     plt.title('Histogram of Heart Rates')
#     plt.xlabel('Heart Rate (BPM)')
#     plt.ylabel('Cumulative Magnitude')
#     plt.show()


def create_animation(data, samples_per_frame, fps, window_size, update_interval):
    fig, ax1, ax2, line1, line2 = setup_plots(1)

    # Create a text artist for peak annotations
    peak_text = ax2.text(0.95, 0.95, '', transform=ax2.transAxes,
                         horizontalalignment='right', verticalalignment='top', color='red')
    frames = np.arange(window_size, len(data), window_size)

    ani1 = FuncAnimation(fig,
                         lambda frame: plot_time_fft(frame, data, samples_per_frame, fps, window_size, line1, line2,
                                                     ax1, ax2, peak_text), frames=frames, blit=False,
                         interval=update_interval * 1000, repeat=False)

    plt.show()

    fig, ax1, ax2, line1, line2 = setup_plots(2)
    ani2 = FuncAnimation(fig,
                         lambda frame: plot_filtered_fft(frame, data, samples_per_frame, fps, window_size,
                                                         line1, line2, ax1, ax2, peak_text), frames=frames,
                         blit=True,
                         interval=update_interval * 1000, repeat=False)
    plt.show()

# plot_histogram(peak_frequencies, peak_magnitudes)