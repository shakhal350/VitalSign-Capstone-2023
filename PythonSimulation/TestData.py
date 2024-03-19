import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c
from scipy.signal import find_peaks

from filter_processing import apply_high_pass_filter, butter_bandpass_filter, lowpass_filter
from SVD_processing import SVD_Matrix
from data_processing import load_and_process_data
from NotBreathing import detect_non_breathing_periods


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


def find_significant_peaks(fft_data, fft_freq, prominence=0.1, width=3, percentile=99):
    # power spectral density (PSD)
    fft_data = np.abs(fft_data) ** 2
    thresh = np.percentile(fft_data, percentile)
    p, prop = find_peaks(fft_data, prominence=prominence, width=width, height=thresh)
    return p, prop


def select_best_peak(p, prop, fft_freq):
    if len(p) > 0:
        scores_select = prop['prominences'] * prop['widths'] * prop['peak_heights']
        best_peak_i = np.argmax(scores_select)
        best_peak_freq = fft_freq[p[best_peak_i]]
    else:
        print("No peaks detected.")
        best_peak_freq = 0
    return best_peak_freq


def determine_frequency_window(best_peak_freq, fft_freq, window_gap=0.1):
    starting = max(best_peak_freq - window_gap, 0)
    ending = best_peak_freq + window_gap
    starting_index = np.where(fft_freq > starting)[0][0]  # Find the index of the frequency closest to the starting value
    ending_index = np.where(fft_freq > ending)[0][0]  # Find the index of the frequency closest to the ending value
    return starting, ending, starting_index, ending_index


def setup_plots(plotnumber, filename=None, filename_truth_Br=None, filename_truth_HR=None, NN=None):
    # pick a random number for window time
    window_time = 60  # <-----*** CHOOSE a window time YOU WANT ***
    random_start = 0
    SVD_order = 4  # <-----*** CHOOSE a SVD order YOU WANT ***

    filename_truth_Br = None
    filename_truth_HR = None

    data_Re, data_Im, radar_parameters = load_and_process_data(filename)
    frameRate = radar_parameters['frameRate']
    samplesPerFrame = radar_parameters['samplesPerFrame']

    data_abs = np.abs(data_Re + 1j * data_Im)
    data_fft, data_FFT_freq = compute_fft(data_abs, frameRate)

    if filename_truth_Br is not None and filename_truth_HR is not None:
        data_BR = np.genfromtxt(filename_truth_Br, delimiter=',')
        data_HR = np.genfromtxt(filename_truth_HR, delimiter=',')

    start = int(random_start * frameRate * samplesPerFrame)
    end = int((random_start + window_time) * frameRate * samplesPerFrame)
    fraction_frame_samples = int(frameRate * samplesPerFrame * 0.1)
    data_Re_small = data_Re[start:start + fraction_frame_samples]
    data_Im_small = data_Im[start:start + fraction_frame_samples]

    data_Re_window = data_Re[start:end]
    data_Im_window = data_Im[start:end]
    ############################################################################################################
    # SVD for noise reduction
    data_Re_SVD = SVD_Matrix(data_Re_window, radar_parameters, SVD_order)
    data_Im_SVD = SVD_Matrix(data_Im_window, radar_parameters, SVD_order)
    filtered_data = data_Re_SVD + 1j * data_Im_SVD

    time_domain = np.linspace(random_start, random_start + window_time, filtered_data.shape[0])
    # FFT of the filtered data
    fft_filtered_data, fft_freq_average = compute_fft(filtered_data, frameRate)

    # Find peaks and their properties
    peaks_average, properties = find_significant_peaks(fft_filtered_data, fft_freq_average)

    # Select the peak with the highest significance score
    best_fft_peak_freq = select_best_peak(peaks_average, properties, fft_freq_average)
    print(f"FFT RANGE: Best Peak Frequency: {best_fft_peak_freq} Hz")
    data_Re_SVD = lowpass_filter(data_Re_SVD, best_fft_peak_freq + 0.2, frameRate, order=6)
    data_Im_SVD = lowpass_filter(data_Im_SVD, best_fft_peak_freq + 0.2, frameRate, order=6)

    # Determine window start and end frequencies around the best peak
    window_start_peak, window_end_peak, window_start_index, window_end_index = determine_frequency_window(best_fft_peak_freq, fft_freq_average)

    phase_values = []
    spectrogram_data = []
    fig_ani, ax = plt.subplots()
    for i in range(0, len(filtered_data), samplesPerFrame):
        # Extract a single frame
        current_frame_Re = data_Re_SVD[i:i + samplesPerFrame]
        current_frame_Im = data_Im_SVD[i:i + samplesPerFrame]
        current_frame = current_frame_Re + 1j * current_frame_Im

        fft_frame, fft_freq_frame = compute_fft(current_frame, frameRate)

        # Find peaks with initial criteria
        percentile_threshold = np.percentile(np.abs(fft_frame), 99)
        peaks, properties = find_peaks(np.abs(fft_frame), width=1, height=percentile_threshold)

        # Find the peak with the highest significance score
        peak_frame, peak_properties = find_significant_peaks(fft_frame, fft_freq_frame, width=1, percentile=99)
        best_range_fft_peak_freq = select_best_peak(peak_frame, peak_properties, fft_freq_frame)

        # if (i % 100) == 0:
        #     ax.clear()
        #     ax.plot(fft_freq_frame, np.abs(fft_frame))
        #     ax.axvline(x=best_range_fft_peak_freq, color='g', linestyle='dotted', label='Center Peak', linewidth=2)
        #     ax.set_title('FFT of Filtered ADC Data')
        #     ax.set_xlabel('Frequency')
        #     ax.set_ylabel('Magnitude')
        #     ax.legend()
        #     clear_output(wait=True)
        #     plt.pause(0.0001)  # Pause briefly to allow the figure to be updated

        # Find the closest fft_freq_frame to the best_range_fft_peak_freq
        best_range_fft_peak_freq_index = np.where(fft_freq_frame > best_range_fft_peak_freq)[0][0]

        spectrogram_data.append(np.abs(fft_frame))  # Append the magnitude of the FFT result for the spectrogram
        phase_values_frame = np.angle(fft_frame[best_range_fft_peak_freq_index])
        phase_values.append(phase_values_frame)  # Append to the overall phase values list
    # plt.close(fig_ani)

    # spectrogram_array = np.array(spectrogram_data)
    # # Plot the spectrogram using matplotlib
    # plt.figure(figsize=(10, 4))
    # plt.imshow(spectrogram_array.T, aspect='auto', origin='lower', extent=[0, len(spectrogram_data), fft_freq_frame[0], fft_freq_frame[-1]])
    # plt.colorbar(label='Magnitude')
    # plt.title('Spectrogram of All Chirps')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Chirp Number')
    # plt.show()

    # Convert phase_values list to a NumPy array
    phase_values_array = np.array(phase_values)

    # Now, you can use the shape attribute
    phase_time = np.linspace(random_start, random_start + window_time, phase_values_array.shape[0])
    unwrap_phase = np.unwrap(phase_values)
    diff_unwrap_phase = np.diff(unwrap_phase)

    print("Unwrapped Phase Values: ")
    print(diff_unwrap_phase[0:10])

    # Assuming diff_unwrap_phase is your unwrapped differential phase array
    threshold = np.std(diff_unwrap_phase) * 2  # Example threshold: 2 times the standard deviation

    # Initialize an array to hold the cleaned phase differences
    cleaned_diff_unwrap_phase = np.copy(diff_unwrap_phase)

    for m in range(1, len(diff_unwrap_phase) - 1):  # Skip the first and last elements for now
        forward_diff = diff_unwrap_phase[m + 1] - diff_unwrap_phase[m]
        backward_diff = diff_unwrap_phase[m] - diff_unwrap_phase[m - 1]

        # Check if either difference exceeds the threshold
        if abs(forward_diff) > threshold or abs(backward_diff) > threshold:
            # Replace with interpolated value
            cleaned_diff_unwrap_phase[m] = (diff_unwrap_phase[m - 1] + diff_unwrap_phase[m + 1]) / 2

    # Handle the first and last elements separately
    forward_diff_first = diff_unwrap_phase[1] - diff_unwrap_phase[0]
    if abs(forward_diff_first) > threshold:
        cleaned_diff_unwrap_phase[0] = diff_unwrap_phase[1]

    lambda_c = 3e8 / radar_parameters['startFreq']
    chest_displacement = ((lambda_c / (4 * np.pi)) * diff_unwrap_phase) * 1000  # in mm
    cleaned_chest_displacement = ((lambda_c / (4 * np.pi)) * cleaned_diff_unwrap_phase) * 1000  # in mm

    # Detect whether the chest displacement is not moving (i.e. the person is not breathing)
    # Apply the function
    non_breathing_periods = detect_non_breathing_periods(unwrap_phase, frameRate, 2, 1.5)

    # Print the results
    print("Non-breathing periods (start_time, end_time in seconds):")
    for start, end in non_breathing_periods:
        duration = end - start
        print(f"From {start:.2f} to {end:.2f} seconds, duration: {duration:.2f} seconds")

    fft_chest_displacement, fft_phase_freq = compute_fft(chest_displacement, frameRate)

    bandpass_chest_displacement_BR = butter_bandpass_filter(cleaned_chest_displacement, 0.1, 0.8, frameRate, order=4)
    bandpass_chest_displacement_HR = butter_bandpass_filter(cleaned_chest_displacement, 0.8, 4.0, frameRate, order=4)

    bandpass_phase_values_BR = butter_bandpass_filter(phase_values, 0.1, 0.8, frameRate, order=4)
    bandpass_phase_values_HR = butter_bandpass_filter(phase_values, 0.8, 4.0, frameRate, order=4)

    bandpass_unwrap_phase_BR = butter_bandpass_filter(unwrap_phase, 0.1, 0.8, frameRate, order=4)
    bandpass_unwrap_phase_HR = butter_bandpass_filter(unwrap_phase, 0.8, 4.0, frameRate, order=4)

    phase_values_fft_BR, phase_freq_fft = compute_fft(bandpass_phase_values_BR, frameRate)
    phase_values_fft_HR, phase_freq_fft = compute_fft(bandpass_phase_values_HR, frameRate)

    unwrap_phase_fft_BR, unwrap_phase_freq = compute_fft(bandpass_unwrap_phase_BR, frameRate)
    unwrap_phase_fft_HR, unwrap_phase_freq = compute_fft(bandpass_phase_values_HR, frameRate)

    fft_band_data_breathing, fft_band_data_breathing_freq = compute_fft(bandpass_chest_displacement_BR, frameRate)
    fft_band_data_cardiac, fft_band_data_cardiac_freq = compute_fft(bandpass_chest_displacement_HR, frameRate)

    best_breathing_freq_peaks, breathing_freq_peaks_properties = find_significant_peaks(fft_band_data_breathing, fft_band_data_breathing_freq, width=1, percentile=99)
    best_cardiac_freq_peaks, cardiac_freq_peaks_properties = find_significant_peaks(fft_band_data_cardiac, fft_band_data_cardiac_freq, width=1, percentile=99)

    # Select the peak with the highest significance score
    best_breathing_freq = select_best_peak(best_breathing_freq_peaks, breathing_freq_peaks_properties, fft_band_data_breathing_freq)
    best_cardiac_freq = select_best_peak(best_cardiac_freq_peaks, cardiac_freq_peaks_properties, fft_band_data_cardiac_freq)

    print(f"Best Breathing Frequency: {best_breathing_freq * 60} BPM")
    print(f"Best Cardiac Frequency: {best_cardiac_freq * 60} BPM")

    if plotnumber == 1:
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        line1, = ax1.plot([], [], lw=1)
        line2, = ax2.plot([], [], lw=1)
        line1.set_data(phase_time, unwrap_phase)
        line2.set_data(phase_time[:-1], cleaned_chest_displacement)

        # ax1.annotate('Best Breathing Frequency = %.2f' % (best_breathing_freq * 60), xy=(0.10, 0.85), xycoords='axes fraction', color='green', fontsize=10, weight='bold')
        ax1.set_title('Breathing Rate')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Magnitude')
        ax1.set_xlim(0, 60)
        ax1.set_ylim(int(np.min(unwrap_phase)) - 1, int(np.max(unwrap_phase)) + 1)

        # ax2.annotate('Best Cardiac Frequency = %.2f' % (best_cardiac_freq * 60), xy=(0.10, 0.85), xycoords='axes fraction', color='green', fontsize=10, weight='bold')
        ax2.set_title('Heart Rate')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Magnitude')
        ax2.set_xlim(0, 60)
        ax2.set_ylim(int(np.min(cleaned_chest_displacement)) - 1, int(np.max(cleaned_chest_displacement)) + 1)

        plt.tight_layout()
        plt.savefig('DefCharts.png')
        return fig1, ax1, ax2, line1, line2, best_breathing_freq * 60, best_cardiac_freq * 60
    elif plotnumber == 2:
        fig2, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 10))
        line1, = ax1.plot([], [], lw=1)
        line2, = ax2.plot([], [], lw=1)
        line3, = ax3.plot([], [], lw=1)
        line4, = ax4.plot([], [], lw=1)
        line5, = ax6.plot([], [], lw=1)
        line6, = ax7.plot([], [], lw=1)
        line7, = ax8.plot([], [], lw=1)

        line1.set_data(data_FFT_freq, data_fft)
        line2.set_data(fft_freq_average, np.abs(fft_filtered_data))
        line3.set_data(phase_time, phase_values)
        line4.set_data(phase_time, unwrap_phase)
        line5.set_data(phase_time[:-1], cleaned_chest_displacement)
        line6.set_data(fft_phase_freq * 60, np.abs(fft_chest_displacement))
        line7.set_data(fft_band_data_breathing_freq * 60, np.abs(fft_band_data_breathing))

        ax1.set_title('FFT of ADC Data')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Magnitude')
        ax1.set_xlim(0, frameRate / 2)
        ax1.set_ylim(0, int(np.max(np.abs(data_fft))) + 1)

        ax2.set_title('FFT of Filtered ADC Data')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Magnitude')
        ax2.set_xlim(0, frameRate / 2)
        ax2.set_ylim(0, int(np.max(np.abs(fft_filtered_data))))

        ax3.set_title('Phase Values')
        ax3.set_xlabel('Time Domain')
        ax3.set_ylabel('Phase')
        ax3.set_xlim(0, 60)
        ax3.set_ylim(int(np.min(phase_values)) - 1, int(np.max(phase_values)) + 1)

        ax4.set_title('Unwrapped Phase Values')
        ax4.set_xlabel('Time Domain')
        ax4.set_ylabel('Phase')
        ax4.set_xlim(0, 60)
        ax4.set_ylim(int(np.min(unwrap_phase)) - 1, int(np.max(unwrap_phase)) + 1)

        ax5.set_title('Chest Displacement from Phase Differencing')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Chest Displacement (cm)')
        ax5.set_xlim(0, 60)
        ax5.set_ylim(int(np.min(chest_displacement)) - 1, int(np.max(chest_displacement)) + 1)

        ax6.set_title('FFT of Chest Displacement')
        ax6.set_xlabel('Frequency (BPM)')
        ax6.set_ylabel('Magnitude')
        ax6.set_xlim(0, 600)
        ax6.set_ylim(0, int(np.max(np.abs(fft_chest_displacement))))

        ax7.set_title('Breathing Region Filtered Data')
        ax7.set_xlabel('Frequency (BPM)')
        ax7.set_ylabel('Magnitude')
        ax7.set_xlim(0, 600)
        ax7.set_ylim(0, int(np.max(np.abs(fft_band_data_breathing))))

        ax8.set_title('Cardiac Region Filtered Data')
        ax8.set_xlabel('Frequency (BPM)')
        ax8.set_ylabel('Magnitude')
        ax8.set_xlim(0, 600)
        ax8.set_ylim(0, int(np.max(np.abs(fft_band_data_cardiac))))

        plt.tight_layout()
        plt.savefig('DevCharts.png')
        return fig2, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, line1, line2, line3, line4, line5, line6, line7, best_breathing_freq * 60, best_cardiac_freq * 60

# # make a big subplot for all the plots
# fig, axs = plt.subplots(4, 3, figsize=(20, 12))
# fig.suptitle('All the plots for file: ' + filename)
# axs[0, 0].plot(data_Re_small[:fraction_frame_samples], label='Real')
# axs[0, 0].plot(data_Im_small[:fraction_frame_samples], label='Imaginary')
# axs[0, 0].set_title('Part of Raw ADC Data Pre filtering')
# axs[0, 0].set_xlabel('Time Domain')
# axs[0, 0].set_ylabel('Amplitude')
# axs[0, 0].legend()
#
# axs[0, 1].plot(data_Re_SVD[:fraction_frame_samples], label='Real')
# axs[0, 1].plot(data_Im_SVD[:fraction_frame_samples], label='Imaginary')
# axs[0, 1].set_title('Part of Filtered ADC Data')
# axs[0, 1].set_xlabel('Time Domain')
# axs[0, 1].set_ylabel('Amplitude')
#
# axs[0, 2].plot(data_FFT_freq, data_fft)
# axs[0, 2].set_title('FFT of ADC Data')
# axs[0, 2].set_xlabel('Frequency')
# axs[0, 2].set_ylabel('Magnitude')
#
# axs[1, 0].axvline(x=window_start_peak, color='k', linestyle='--', label='Window Start/End')
# axs[1, 0].axvline(x=window_end_peak, color='k', linestyle='--')
# axs[1, 0].axvline(x=best_range_fft_peak_freq, color='g', linestyle='dotted', label='Center Peak', linewidth=2)
# axs[1, 0].plot(fft_freq_average, np.abs(fft_filtered_data))
# axs[1, 0].plot(fft_freq_average[peaks_average], np.abs(fft_filtered_data)[peaks_average], "o", label='peaks', color='r', markersize=2)
# axs[1, 0].set_title('FFT of Filtered ADC Data')
# axs[1, 0].set_xlabel('Frequency')
# axs[1, 0].set_ylabel('Magnitude')
# axs[1, 0].legend()
#
# axs[1, 1].plot(phase_time, phase_values)
# axs[1, 1].set_title('Phase Values')
# axs[1, 1].set_xlabel('Time Domain')
# axs[1, 1].set_ylabel('Phase')
#
# axs[1, 2].plot(phase_time, unwrap_phase)
# axs[1, 2].set_title('Unwrapped Phase Values')
# axs[1, 2].set_xlabel('Time Domain')
# axs[1, 2].set_ylabel('Phase')
#
# axs[2, 0].plot(phase_time[:-1], chest_displacement, "r")
# axs[2, 0].plot(phase_time[:-1], cleaned_chest_displacement, "g")
# axs[2, 0].set_title('Chest Displacement from Phase Differencing')
# axs[2, 0].set_xlabel('Time')
# axs[2, 0].set_ylabel('Chest Displacement (cm)')
#
# axs[2, 1].plot(fft_phase_freq * 60, (np.abs(fft_chest_displacement)))
# axs[2, 1].set_title('FFT of Chest Displacement')
# axs[2, 1].set_xlabel('Frequency (BPM)')
# axs[2, 1].set_ylabel('Magnitude')
#
# axs[2, 2].plot(fft_band_data_breathing_freq * 60, np.abs(fft_band_data_breathing))
# axs[2, 2].axvline(x=best_breathing_freq * 60, color='g', linestyle='dotted', label='Breathing Frequency', linewidth=2)
# axs[2, 2].annotate('Best Breathing Frequency = %.2f' % (best_breathing_freq * 60), xy=(0.30, 0.85), xycoords='axes fraction', color='green', fontsize=14, weight='bold')
# axs[2, 2].set_title('Breathing Region Filtered Data')
# axs[2, 2].set_xlabel('Frequency (BPM)')
# axs[2, 2].set_ylabel('Magnitude')
#
# axs[3, 0].plot(fft_band_data_cardiac_freq * 60, np.abs(fft_band_data_cardiac))
# axs[3, 0].axvline(x=best_cardiac_freq * 60, color='g', linestyle='dotted', label='Cardiac Frequency', linewidth=2)
# axs[3, 0].annotate('Best Cardiac Frequency = %.2f' % (best_cardiac_freq * 60), xy=(0.30, 0.85), xycoords='axes fraction', color='green', fontsize=14, weight='bold')
# axs[3, 0].set_title('Cardiac Region Filtered Data')
# axs[3, 0].set_xlabel('Frequency (BPM)')
# axs[3, 0].set_ylabel('Magnitude')
#
# if filename_truth_Br is not None and filename_truth_HR is not None:
#     ground_truth_time = np.linspace(random_start, random_start + window_time, data_BR[random_start: random_start + window_time].shape[0])
#     axs[3, 1].plot(ground_truth_time, data_BR[random_start: random_start + window_time])
#     axs[3, 1].set_title('Breathing Rate')
#     axs[3, 1].set_xlabel('Time')
#     axs[3, 1].set_ylabel('Rate')
#     axs[3, 1].legend(['Breathing Rate'])
#     # Calculate the average Heart Rate
#     average_Br = np.mean(data_BR[random_start: random_start + window_time])
#     # Annotate the average Heart Rate
#     axs[3, 1].annotate('Average BR = %.2f' % average_Br, xy=(0.30, 0.85), xycoords='axes fraction', color='green', fontsize=14, weight='bold')
#
#     axs[3, 2].plot(ground_truth_time, data_HR[random_start: random_start + window_time])
#     axs[3, 2].set_title('Heart Rate')
#     axs[3, 2].set_xlabel('Time')
#     axs[3, 2].set_ylabel('Rate')
#     axs[3, 2].legend(['Heart Rate'])
#     # Calculate the average Heart Rate
#     average_HR = np.mean(data_HR[random_start: random_start + window_time])
#     # Annotate the average Heart Rate
#     axs[3, 2].annotate('Average HR = %.2f' % average_HR, xy=(0.30, 0.85), xycoords='axes fraction', color='red', fontsize=14, weight='bold')
#
# # fig2, ax2 = plt.subplots(2, 2)
# # ax2[0, 0].plot(phase_freq_fft * 60, np.abs(phase_values_fft_BR))
# # ax2[0, 0].set_title('FFT of Phase Values in Breathing Region')
# # ax2[0, 0].set_xlabel('BPM')
# # ax2[0, 0].set_ylabel('Magnitude')
# #
# # ax2[0, 1].plot(phase_freq_fft * 60, np.abs(phase_values_fft_HR))
# # ax2[0, 1].set_title('FFT of Phase Values in Cardiac Region')
# # ax2[0, 1].set_xlabel('BPM')
# # ax2[0, 1].set_ylabel('Magnitude')
# #
# # ax2[1, 0].plot(unwrap_phase_freq * 60, np.abs(unwrap_phase_fft_BR))
# # ax2[1, 0].set_title('FFT of Unwrapped Phase Values in Breathing Region')
# # ax2[1, 0].set_xlabel('BPM')
# # ax2[1, 0].set_ylabel('Magnitude')
# #
# # ax2[1, 1].plot(unwrap_phase_freq * 60, np.abs(unwrap_phase_fft_HR))
# # ax2[1, 1].set_title('FFT of Unwrapped Phase Values in Cardiac Region')
# # ax2[1, 1].set_xlabel('BPM')
# # ax2[1, 1].set_ylabel('Magnitude')
#
# plt.tight_layout()
# plt.show()
#
# print("done")
