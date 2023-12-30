import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from fft_processing import perform_fft, apply_magnitude_cutoff, perform_ifft


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


def update_plot(frame, data, fps, window_size, cutoff_threshold, line1, line2, line3, line4, ax1, ax2, ax3, ax4):
    start = max(0, frame - window_size * fps)
    end = frame

    time_data = np.abs(data.iloc[start:end, 0])
    line1.set_data(np.arange(start, end), time_data)
    ax1.set_xlim(start, end)
    ax1.set_ylim(np.min(time_data), np.max(time_data))

    # Call FFT processing functions
    fft_freq, fft_magnitude = perform_fft(time_data, fps)
    line2.set_data(fft_freq[:len(fft_magnitude)//2],
                   fft_magnitude[:len(fft_magnitude)//2])
    ax2.set_xlim(0, fps / 2)
    ax2.set_ylim(0, np.max(fft_magnitude))

    fft_magnitude_cutoff = apply_magnitude_cutoff(
        fft_magnitude, cutoff_threshold)
    line3.set_data(fft_freq[:len(fft_magnitude_cutoff)//2],
                   fft_magnitude_cutoff[:len(fft_magnitude_cutoff)//2])
    ax3.set_xlim(0, fps / 2)
    ax3.set_ylim(0, np.max(fft_magnitude_cutoff))

    cutoff_fft_result = perform_ifft(fft_magnitude_cutoff)
    line4.set_data(np.arange(start, end), np.abs(cutoff_fft_result))
    ax4.set_xlim(start, end)
    ax4.set_ylim(0, np.max(np.abs(cutoff_fft_result)))

    return line1, line2, line3, line4


def create_animation(fig, data, fps, window_size, update_interval, cutoff_threshold, line1, line2, line3, line4, ax1, ax2, ax3, ax4):
    ani = FuncAnimation(fig, lambda frame: update_plot(frame, data, fps, window_size, cutoff_threshold, line1, line2, line3, line4, ax1, ax2, ax3, ax4),
                        frames=np.arange(window_size * fps, len(data)), blit=True, interval=update_interval * 1000)
    plt.show()
