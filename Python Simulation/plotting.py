import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from fft_processing import perform_fft, apply_magnitude_cutoff
from scipy.signal import find_peaks

# Peak data to global lists
global peak_frequencies, peak_magnitudes
peak_frequencies = []
peak_magnitudes = []


def setup_plots():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    line1, = ax1.plot([], [], lw=2)
    line2, = ax2.plot([], [], lw=2)
    line3, = ax3.plot([], [], lw=2)
    line4, = ax4.plot([], [], lw=2)

    ax1.set_title('Magnitude of Signal Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Magnitude')

    ax2.set_title('Untouched FFT of Signal')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')

    ax3.set_title('FFT of Signal WITH Cutoff')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')

    ax4.set_title('Reconstructed Signal from Cutoff FFT')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Magnitude')

    plt.tight_layout()
    return fig, ax1, ax2, ax3, ax4, line1, line2, line3, line4


def update_plot(current_sample, data, samples_per_frame, fps, sample_window_size, cutoff_threshold, line1, line2, line3,
                line4, ax1,
                ax2, ax3, ax4,
                peak_text):
    starting_sample = max(0, current_sample - sample_window_size)
    ending_sample = current_sample

    sample_per_second = samples_per_frame * fps  # 512samples/frames * 20frames/sec = 10240 samples/frame
    min_freq_cutoff = 0.03

    latest_time = ending_sample / sample_per_second  # Time of the latest data point
    time_data = np.abs(data.iloc[starting_sample:ending_sample])
    # Calculate the time values for the x-axis
    time_values = np.linspace(starting_sample / sample_per_second, ending_sample / sample_per_second, num=len(time_data))
    print(f"Current Sample #{current_sample} at Time: {latest_time} seconds")
    # Set the data for the line plot with time_values as x-axis and time_data as y-axis
    line1.set_data(time_values, time_data)
    ax1.set_xlim(starting_sample / sample_per_second, ending_sample / sample_per_second)
    ax1.set_ylim(np.min(time_data), np.max(time_data))

    # Call FFT processing functions
    fft_freq, fft_magnitude = perform_fft(time_data, 1)
    line2.set_data(fft_freq[:len(fft_magnitude) // 2], fft_magnitude[:len(fft_magnitude) // 2])
    ax2.set_xlim(0, max(fft_freq))
    ax2.set_ylim(0, np.max(fft_magnitude[np.argmax(fft_freq >= min_freq_cutoff):]))  # ignoring first 0.03Hz

    fft_magnitude_cutoff = apply_magnitude_cutoff(
        fft_magnitude, cutoff_threshold)
    line3.set_data(fft_freq[:len(fft_magnitude_cutoff) // 2], fft_magnitude_cutoff[:len(fft_magnitude_cutoff) // 2])
    ax3.set_xlim(0, max(fft_freq))
    ax3.set_ylim(0, np.max(fft_magnitude[np.argmax(fft_freq >= min_freq_cutoff):]))

    # Filter out frequencies below min_freq_cutoff Hz
    min_freq_index = np.argmax(fft_freq >= min_freq_cutoff)
    fft_freq_positive = fft_freq[min_freq_index:len(fft_magnitude_cutoff) // 2]
    fft_magnitude_cutoff_positive = fft_magnitude_cutoff[min_freq_index:len(fft_magnitude_cutoff) // 2]

    # Find peaks in the positive side of the cutoff FFT
    peaks, _ = find_peaks(fft_magnitude_cutoff_positive)

    # Pair peak frequencies with their magnitudes and sort by magnitude (descending)
    peak_freqs_magnitudes = [(fft_freq[peak + min_freq_index], fft_magnitude_cutoff_positive[peak]) for peak in peaks]
    peak_freqs_magnitudes.sort(key=lambda x: x[1], reverse=True)

    # Convert frequencies to BPM and create annotation text
    peak_freq_bpm_text = '\n'.join(f"{freq:.2f} Hz = {freq * 60:.2f} BPM" for freq, mag in peak_freqs_magnitudes)
    peak_text.set_text(peak_freq_bpm_text)

    cutoff_fft_result = np.fft.ifft(fft_magnitude)
    line4.set_data(time_values, np.abs(cutoff_fft_result))
    ax4.set_xlim(starting_sample / sample_per_second, ending_sample / sample_per_second)
    ax4.set_ylim(0, np.max(np.abs(cutoff_fft_result)))

    for freq, mag in peak_freqs_magnitudes:
        peak_frequencies.append(freq)
        peak_magnitudes.append(mag)

    return line1, line2, line3, line4, peak_text


# After animation code
def plot_histogram(peak_frequencies, peak_magnitudes):
    # Convert frequencies to heart rates in BPM
    heart_rates = [freq * 60 for freq in peak_frequencies]

    # Create a histogram of heart rates weighted by their magnitudes
    plt.figure()
    plt.hist(heart_rates, weights=peak_magnitudes, bins=range(int(min(heart_rates)), int(max(heart_rates)) + 1, 1))
    plt.title('Histogram of Heart Rates')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Cumulative Magnitude')
    plt.show()


def create_animation(fig, data, samples_per_frame, fps, window_size, update_interval, cutoff_threshold, line1, line2,
                     line3, line4, ax1,
                     ax2, ax3, ax4):
    # Create a text artist for peak annotations
    peak_text = ax3.text(0.95, 0.95, '', transform=ax3.transAxes, horizontalalignment='right', verticalalignment='top',
                         color='red')

    frames = np.arange(window_size, len(data), window_size)
    # print("List of frames:", frames)

    ani = FuncAnimation(fig,
                        lambda frame: update_plot(frame, data, samples_per_frame, fps, window_size, cutoff_threshold,
                                                  line1, line2, line3,
                                                  line4, ax1, ax2, ax3, ax4, peak_text), frames=frames,
                        blit=False,
                        interval=update_interval * 1000)
    plt.show()
    plot_histogram(peak_frequencies, peak_magnitudes)
