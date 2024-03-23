import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from processing_utils import apply_windowing, compute_fft, find_significant_peaks, select_best_peak, determine_frequency_window
from data_processing import load_and_process_data
from SVD_processing import SVD_Matrix


# Example utility function file (processing_utils.py) includes functions like apply_windowing, compute_fft, etc.

def process_data_and_plot(filename, frameRate, samplesPerFrame, random_start=10, window_time=60, SVD_order=100):
    """
    Main function to process the radar data and plot the results.
    - filename: Path to the dataset.
    - frameRate, samplesPerFrame: Radar parameters.
    - random_start: Start time for processing.
    - window_time: Window duration for signal processing.
    - SVD_order: Order for Singular Value Decomposition.
    """
    # Load and preprocess data
    data_Re, data_Im, radar_parameters = load_and_process_data(filename)

    # Convert to absolute and compute FFT
    data_abs = np.abs(data_Re + 1j * data_Im)
    data_fft, data_FFT_freq = compute_fft(data_abs, frameRate)

    # Data preparation for small and windowed segments
    data_Re_small, data_Im_small, data_Re_window, data_Im_window = prepare_data_for_plots(data_Re, data_Im, random_start, window_time, frameRate, samplesPerFrame)

    # SVD for noise reduction
    data_Re_SVD, data_Im_SVD = SVD_Matrix(data_Re_window, data_Im_window, radar_parameters, SVD_order)

    # Plotting the results
    plot_data(data_Re_small, data_Im_small, fraction_frame_samples, 'Part of Raw ADC Data Pre filtering')
    plot_data(data_Re_SVD, data_Im_SVD, fraction_frame_samples, 'Part of Filtered ADC Data')
    plot_fft(data_FFT_freq, data_fft, 'FFT of ADC Data')

    # Phase and Unwrapped Phase Processing and Plotting
    phase_values, phase_time, fft_filtered_data, fft_freq_average, peaks_average, properties = plot_phase(data_Re_SVD, data_Im_SVD, samplesPerFrame, frameRate, random_start, window_time)
    unwrap_phase, corrected_phase, estimated_baseline = plot_unwrap_phase(phase_values, phase_time)

    # Displacement Calculation and Plotting
    chest_displacement, cleaned_chest_displacement = plot_displacement(unwrap_phase, radar_parameters, frameRate)

    # Spectrogram and Breathing/Heart Rate Analysis
    plot_spectrogram(filtered_data, samplesPerFrame, frameRate)
    plot_heart_breathing_rate(fft_band_data_breathing_freq, fft_band_data_breathing, best_breathing_freq, fft_band_data_cardiac_freq, fft_band_data_cardiac, best_cardiac_freq)

    plt.tight_layout()
    plt.show()
    print("Processing and plotting complete.")

filename = "path/to/dataset.csv"
dataRe, dataIm, radar_parameters = load_and_process_data(filename)
frameRate = radar_parameters["frameRate"]
samplesPerFrame = radar_parameters["samplesPerFrame"]
process_data_and_plot(filename, frameRate, samplesPerFrame, random_start=10, window_time=60, SVD_order=100)
