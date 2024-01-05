import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from fft_processing import perform_fft, apply_magnitude_cutoff, perform_ifft
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


def update_plot(frame, data, fps, window_size, cutoff_threshold, line1, line2, line3, line4, ax1, ax2, ax3, ax4,
                peak_text):
    start = max(0, frame - window_size)
    end = frame
    min_freq_cutoff = 0.03

    latest_time = end / 10000  # Time of the latest data point
    time_data = (np.abs(data.iloc[start:end]))
    print(f"Frame #{frame} at Time: {latest_time} seconds")
    # print(f"Time domain Datapoints: \n{time_data}")
    line1.set_data(np.arange(start, end), time_data)
    ax1.set_xlim(start, end)
    ax1.set_ylim(np.min(time_data), np.max(time_data))

    # Call FFT processing functions
    fft_freq, fft_magnitude = perform_fft(time_data, fps)
    line2.set_data(fft_freq[:len(fft_magnitude) // 2],
                   fft_magnitude[:len(fft_magnitude) // 2])
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, np.max(fft_magnitude[np.argmax(fft_freq >= min_freq_cutoff):]))  # ignoring first 0.03Hz

    fft_magnitude_cutoff = apply_magnitude_cutoff(
        fft_magnitude, cutoff_threshold)
    line3.set_data(fft_freq[:len(fft_magnitude_cutoff) // 2],
                   fft_magnitude_cutoff[:len(fft_magnitude_cutoff) // 2])
    ax3.set_xlim(0, 3)
    ax3.set_ylim(0, np.max(fft_magnitude[np.argmax(fft_freq >= min_freq_cutoff):]))

    # Filter out frequencies below min_freq_cutoff Hz
    min_freq_index = np.argmax(fft_freq >= min_freq_cutoff)
    fft_freq_positive = fft_freq[min_freq_index:len(fft_magnitude_cutoff) // 2]
    fft_magnitude_cutoff_positive = fft_magnitude_cutoff[min_freq_index:len(fft_magnitude_cutoff) // 2]

    # Find peaks in the positive side of the cutoff FFT
    peaks, _ = find_peaks(fft_magnitude_cutoff_positive)

    # Adjust peak indices to account for the filtered range and calculate BPM
    peak_freqs = fft_freq[peaks + min_freq_index]
    peak_bpm = peak_freqs * 60  # Convert frequencies to BPM

    # Pair peak frequencies with their magnitudes and sort by magnitude (descending)
    peak_freqs_magnitudes = [(fft_freq[peak + min_freq_index], fft_magnitude_cutoff_positive[peak]) for peak in peaks]
    peak_freqs_magnitudes.sort(key=lambda x: x[1], reverse=True)

    # Convert frequencies to BPM and create annotation text
    peak_freq_bpm_text = '\n'.join(f"{freq:.2f} Hz = {freq * 60:.2f} BPM" for freq, mag in peak_freqs_magnitudes)
    peak_text.set_text(peak_freq_bpm_text)

    cutoff_fft_result = perform_ifft(fft_magnitude_cutoff)
    line4.set_data(np.arange(start, end), np.abs(cutoff_fft_result))
    ax4.set_xlim(start, end)
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


def create_animation(fig, data, fps, window_size, update_interval, cutoff_threshold, line1, line2, line3, line4, ax1,
                     ax2, ax3, ax4):

    # Create a text artist for peak annotations
    peak_text = ax3.text(0.95, 0.95, '', transform=ax3.transAxes, horizontalalignment='right', verticalalignment='top',
                         color='red')

    frames = np.arange(window_size, len(data), window_size)
    # print("List of frames:", frames)

    ani = FuncAnimation(fig,
                        lambda frame: update_plot(frame, data, fps, window_size, cutoff_threshold, line1, line2, line3,
                                                  line4, ax1, ax2, ax3, ax4, peak_text), frames=frames,
                        blit=True,
                        interval=update_interval * 1000)
    plt.show()
    plot_histogram(peak_frequencies, peak_magnitudes)
