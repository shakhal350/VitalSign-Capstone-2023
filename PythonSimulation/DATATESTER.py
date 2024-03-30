import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

import os
import random

from SVD_processing import SVD_Matrix
from personDetection import detect_person_by_svd
from filter_processing import highpass_filter, bandpass_filter, lowpass_filter, bandstop_filter
from data_processing import load_and_process_data
from NotBreathing import detect_non_breathing_periods


def pick_random_file_from_subfolders(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    # Check if there are any files collected
    if all_files:
        # Randomly pick a file
        random_file = random.choice(all_files)
        return random_file
    else:
        return None


def apply_windowing(time_signal):
    # Hamming window
    window = np.hamming(len(time_signal))
    return time_signal * window


def convert_Hz_to_Range(f_beat, B, c=3e8):
    # Calculate the range bins
    range_bins = (c * f_beat) / (2 * B)
    return range_bins


def convert_Range_to_Hz(range_bins, B, c=3e8):
    # Calculate the frequency bins
    f_beat = (2 * range_bins * B) / c
    return f_beat


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


def find_significant_peaks(fft_data, prominence=0.1, width=3, percentile=95):
    # Check if fft_data is complex and compute PSD
    psd_data = np.abs(fft_data)**2 if np.iscomplexobj(fft_data) else fft_data**2
    
    # Find the threshold based on the percentile of the PSD data
    thresh = np.percentile(psd_data, percentile)
    
    # Find peaks using the calculated threshold and specified parameters
    peaks, properties = find_peaks(psd_data, prominence=prominence, width=width, height=thresh)
    
    return peaks, properties


def estimate_peak_intervals(peaks, fft_freqs, fs):
    # Ensure there are at least two peaks to calculate intervals
    if len(peaks) > 1:
        # Calculate the frequency differences between each consecutive peak
        peak_freq_diffs = np.diff(fft_freqs[peaks])
        
        # Convert the frequency intervals to periods (1/frequency)
        peak_periods = 1 / peak_freq_diffs
        
        # Estimate the rate as the median of the calculated periods
        peak_rate = np.median(peak_periods)
        
        # Convert rate to Hz if needed
        peak_rate_hz = peak_rate * fs
        
        return peak_periods, peak_rate_hz
    else:
        return np.array([]), 0




def select_best_peak(p, prop, fft_freq):
    if len(p) > 0:
        scores_select = prop['prominences'] * prop['widths'] * prop['peak_heights']
        best_peak_i = np.argmax(scores_select)
        best_peak_freq = fft_freq[p[best_peak_i]]
    else:
        print("No peaks detected.")
        best_peak_freq = 0
    return best_peak_freq


def determine_frequency_window(best_peak_freq, fft_freq, window_gap=0.2):
    starting = max(best_peak_freq - window_gap, 0.1)
    ending = best_peak_freq + window_gap
    starting_index = np.where(fft_freq > starting)[0][0]  # Find the index of the frequency closest to the starting value
    ending_index = np.where(fft_freq > ending)[0][0]  # Find the index of the frequency closest to the ending value
    return starting, ending, starting_index, ending_index


start_time = 0
sampleNumber = np.random.randint(1, 50)  # <-----*** Randomly picks a sample number *** MAKE SURE TO CHANGE THIS TO A SPECIFIC NUMBER IF YOU WANT TO TEST A SPECIFIC FILE
# sampleNumber = 41
filename_truth_Br = None
filename_truth_HR = None
# filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_' + str(sampleNumber) + '.csv'
# filename_truth_Br = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Heart Rate & Breathing Rate\Breath_' + str(sampleNumber) + '.csv'
# filename_truth_HR = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Heart Rate & Breathing Rate\Heart_' + str(sampleNumber) + '.csv'
# filename = r"..\\PythonSimulation\\Dataset\\DCA1000EVM_Shayan_19Br_100Hr.csv"
# filename = r"C:\Users\Shaya\OneDrive - Concordia University - Canada\UNIVERSITY\CAPSTONE\Our Datasets (DCA1000EVM)\1443_DATASET\Joseph\1m_Data_face\DCA1000EVM_Joseph_15br_65_hr.csv"

folder_path = r"C:\Users\Shaya\OneDrive - Concordia University - Canada\UNIVERSITY\CAPSTONE\Our Datasets (DCA1000EVM)\1443_DATASET"
filename = pick_random_file_from_subfolders(folder_path)
if filename:
    print(f"Randomly selected file: {filename}")
else:
    print("The folder and its subfolders are empty or do not contain any files.")


data_Re, data_Im, radar_parameters = load_and_process_data(filename)
frameRate = radar_parameters['frameRate']
samplesPerFrame = radar_parameters['samplesPerFrame']

# window_time = int(len(data_Re) / (frameRate * samplesPerFrame))
window_time = 60

if filename_truth_Br is not None and filename_truth_HR is not None:
    data_BR = np.genfromtxt(filename_truth_Br, delimiter=',')
    data_HR = np.genfromtxt(filename_truth_HR, delimiter=',')

start = int(start_time * frameRate * samplesPerFrame)
end = int((start_time + window_time) * frameRate * samplesPerFrame)

data_Re_window = data_Re[start:end]
data_Im_window = data_Im[start:end]
filtered_data = data_Re_window + 1j * data_Im_window
time_domain = np.linspace(start_time, start_time + window_time, filtered_data.shape[0])
######################################## FFT of the filtered data ########################################
spectrum_detect = detect_person_by_svd(filtered_data, radar_parameters, 150)
if spectrum_detect is not False:
    print("***Static noise Detected***")
    print(f":Static noise at {spectrum_detect} Hz")
    bandpassgap = 0.1
    data_Re_window = bandstop_filter(data_Re_window, spectrum_detect - bandpassgap, spectrum_detect + bandpassgap, frameRate, order=4)
    data_Im_window = bandstop_filter(data_Im_window, spectrum_detect - bandpassgap, spectrum_detect + bandpassgap, frameRate, order=4)
    filtered_data = data_Re_window + 1j * data_Im_window
else:
    print("***No Static noise Detected***")

# FFT of the filtered data
fft_filtered_data, fft_freq_average = compute_fft(filtered_data, frameRate)

# Find peaks and their properties
peaks_average, properties = find_significant_peaks(fft_filtered_data, width=3, percentile=90)

# Select the peak with the highest significance score
best_fft_peak_freq = select_best_peak(peaks_average, properties, fft_freq_average)
print(f"FFT RANGE: Best Peak Frequency: {best_fft_peak_freq} Hz")

# Determine window start and end frequencies around the best peak
window_start_peak, window_end_peak, window_start_index, window_end_index = determine_frequency_window(best_fft_peak_freq, fft_freq_average)
# round to the nearest decimal
window_start_peak = round(window_start_peak, 3)
window_end_peak = round(window_end_peak, 3)
print(f"Window Start Frequency: {window_start_peak} Hz, Window End Frequency: {window_end_peak} Hz")

data_Re_window = bandpass_filter(filtered_data, window_start_peak, window_end_peak, frameRate, order=4)
data_Im_window = bandpass_filter(filtered_data, window_start_peak, window_end_peak, frameRate, order=4)

data_Re_window = SVD_Matrix(np.real(data_Re_window), radar_parameters, 8)
data_Im_window = SVD_Matrix(np.imag(data_Im_window), radar_parameters, 8)

######################################## Phase and Unwrapped Phase Processing ########################################
phase_values = []
spectrogram_data = []
fig_ani, ax = plt.subplots()
for i in range(0, len(filtered_data), samplesPerFrame):
    # Extract a single frame
    current_frame_Re = data_Re_window[i:i + samplesPerFrame]
    current_frame_Im = data_Im_window[i:i + samplesPerFrame]
    current_frame = current_frame_Re + 1j * current_frame_Im

    fft_frame, fft_freq_frame = compute_fft(current_frame, frameRate)

    # Find peaks with initial criteria
    percentile_threshold = np.percentile(np.abs(fft_frame), 99)
    peaks, properties = find_peaks(np.abs(fft_frame), width=1, height=percentile_threshold)

    # Find the peak with the highest significance score
    peak_frame, peak_properties = find_significant_peaks(fft_frame, width=1, percentile=99)
    best_range_fft_peak_freq = select_best_peak(peak_frame, peak_properties, fft_freq_frame)

    if (i % 100) == 0:  # Update the plot every 50 frames
        ax.clear()
        ax.plot(fft_freq_frame, np.abs(fft_frame))
        ax.axvline(x=best_range_fft_peak_freq, color='g', linestyle='dotted', label='Center Peak', linewidth=2)
        ax.set_title('FFT of Filtered ADC Data')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude')
        ax.legend()
        plt.pause(0.00001)  # Pause briefly to allow the figure to be updated

    # Find the closest fft_freq_frame to the best_range_fft_peak_freq
    best_range_fft_peak_freq_index = np.where(fft_freq_frame > best_range_fft_peak_freq)[0][0]

    spectrogram_data.append(np.abs(fft_frame))  # Append the magnitude of the FFT result for the spectrogram
    phase_values_frame = np.angle(fft_frame[best_range_fft_peak_freq_index])
    phase_values.append(phase_values_frame)  # Append to the overall phase values list
plt.close(fig_ani)
spectrogram_array = np.array(spectrogram_data)

######################################## Unwrapped Phase Processing ########################################
# Convert phase_values list to a NumPy array
phase_values_array = np.array(phase_values)
# Now, you can use the shape attribute
phase_time = np.linspace(start_time, start_time + window_time, phase_values_array.shape[0])
unwrap_phase = np.unwrap(phase_values)
polynomial_coefficients = np.polyfit(phase_time, unwrap_phase, 10)
estimated_baseline = np.polyval(polynomial_coefficients, phase_time)
corrected_phase = unwrap_phase - estimated_baseline

######################################## Cleaning the phase differences ########################################
diff_unwrap_phase = np.diff(unwrap_phase)
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

############################################ Chest Displacement Calculation ############################################
lambda_c = 3e8 / radar_parameters['startFreq']
chest_displacement = ((lambda_c / (4 * np.pi)) * diff_unwrap_phase) * 1000  # in mm
cleaned_chest_displacement = ((lambda_c / (4 * np.pi)) * cleaned_diff_unwrap_phase) * 1000  # in mm

######################################## Detecting Non-Breathing Periods ########################################
# Detect whether the chest displacement is not moving (i.e. the person is not breathing)
# Apply the function
non_breathing_periods = detect_non_breathing_periods(unwrap_phase, frameRate, 4, 1.5)
# Print the results
if len(non_breathing_periods) == 0:
    print("No non-breathing periods detected.")
else:
    print(f"Non-breathing periods detected: {non_breathing_periods}")
for start, end in non_breathing_periods:
    duration = end - start
    print(f"From {start:.2f} to {end:.2f} seconds, duration: {duration:.2f} seconds")
if len(non_breathing_periods) > 0:
    print(f"Total duration of non-breathing periods: {np.sum([end - start for start, end in non_breathing_periods]):.2f} seconds")

######################################## FFT of the chest displacement ########################################
fft_chest_displacement, fft_phase_freq = compute_fft(chest_displacement, frameRate)

bandpass_chest_displacement_BR = bandpass_filter(cleaned_chest_displacement, 0.1, 0.8, frameRate, order=4)
bandpass_chest_displacement_HR = bandpass_filter(cleaned_chest_displacement, 0.8, 4.0, frameRate, order=4)

fft_band_data_breathing, fft_band_data_breathing_freq = compute_fft(bandpass_chest_displacement_BR, frameRate)
fft_band_data_cardiac, fft_band_data_cardiac_freq = compute_fft(bandpass_chest_displacement_HR, frameRate)

best_breathing_freq_peaks, breathing_freq_peaks_properties = find_significant_peaks(fft_band_data_breathing, width=1, percentile=99)
best_cardiac_freq_peaks, cardiac_freq_peaks_properties = find_significant_peaks(fft_band_data_cardiac, width=1, percentile=99)

# After finding significant peaks for the breathing band
breathing_peak_intervals, breathing_peak_rate_hz = estimate_peak_intervals(
    best_breathing_freq_peaks, 
    fft_band_data_breathing_freq, 
    frameRate
)

# After finding significant peaks for the cardiac band
cardiac_peak_intervals, cardiac_peak_rate_hz = estimate_peak_intervals(
    best_cardiac_freq_peaks, 
    fft_band_data_cardiac_freq, 
    frameRate
)

# Convert rates from Hz to BPM and print them
breathing_peak_rate_bpm = breathing_peak_rate_hz * 60
cardiac_peak_rate_bpm = cardiac_peak_rate_hz * 60

print(f"\nBreathing Peak Intervals: {breathing_peak_intervals}")
print(f"Breathing Median Peak Rate: {breathing_peak_rate_bpm} BPM")

print(f"\nCardiac Peak Intervals: {cardiac_peak_intervals}")
print(f"Cardiac Median Peak Rate: {cardiac_peak_rate_bpm} BPM")

# Select the peak with the highest significance score
best_breathing_freq = select_best_peak(best_breathing_freq_peaks, breathing_freq_peaks_properties, fft_band_data_breathing_freq)
best_cardiac_freq = select_best_peak(best_cardiac_freq_peaks, cardiac_freq_peaks_properties, fft_band_data_cardiac_freq)


# print(f"Best Breathing Frequency: {best_breathing_freq * 60} BPM")
# print(f"Best Cardiac Frequency: {best_cardiac_freq * 60} BPM")

# make a big subplot for all the plots
fig, axs = plt.subplots(2, 4, figsize=(30, 8))
fig.suptitle('All the plots for file: ' + filename)
axs[0, 0].imshow(spectrogram_array, aspect='auto', origin='lower', extent=[fft_freq_frame[0], fft_freq_frame[-1], 0, len(spectrogram_data)])
axs[0, 0].set_title('Spectrogram of All Chirps')
axs[0, 0].set_xlabel('Frequency (Hz)')
axs[0, 0].set_ylabel('Chirp Number')

axs[0, 1].axvline(x=window_start_peak, color='k', linestyle='--', label='Window Start/End')
axs[0, 1].axvline(x=window_end_peak, color='k', linestyle='--')
axs[0, 1].axvline(x=best_range_fft_peak_freq, color='g', linestyle='dotted', label='Center Peak', linewidth=2)
axs[0, 1].plot(fft_freq_average, np.abs(fft_filtered_data))
axs[0, 1].plot(fft_freq_average[peaks_average], np.abs(fft_filtered_data)[peaks_average], "o", label='peaks', color='r', markersize=2)
axs[0, 1].set_title('FFT of Filtered ADC Data')
axs[0, 1].set_xlabel('Frequency')
axs[0, 1].set_ylabel('Magnitude')
axs[0, 1].legend()

axs[0, 2].plot(phase_time, phase_values)
axs[0, 2].set_title('Phase Values')
axs[0, 2].set_xlabel('Time Domain')
axs[0, 2].set_ylabel('Phase')

axs[0, 3].plot(phase_time, unwrap_phase, label='Unwrapped Phase', color='r', linewidth=0.5)
axs[0, 3].plot(phase_time, estimated_baseline, label='Estimated Baseline', color='b', linewidth=0.5)
axs[0, 3].plot(phase_time, corrected_phase, label='Baseline Corrected Phase', color='g', linewidth=1)
axs[0, 3].set_title('Unwrapped Phase Values')
axs[0, 3].set_xlabel('Time Domain')
axs[0, 3].set_ylabel('Phase')

axs[1, 0].plot(phase_time[:-1], chest_displacement, "r")
axs[1, 0].plot(phase_time[:-1], cleaned_chest_displacement, "g")
axs[1, 0].set_title('Chest Displacement from Phase Differencing')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Chest Displacement (cm)')

axs[1, 1].plot(fft_phase_freq * 60, (np.abs(fft_chest_displacement)))
axs[1, 1].set_title('FFT of Chest Displacement')
axs[1, 1].set_xlabel('Frequency (BPM)')
axs[1, 1].set_ylabel('Magnitude')

axs[1, 2].plot(fft_band_data_breathing_freq * 60, np.abs(fft_band_data_breathing))
axs[1, 2].axvline(x=best_breathing_freq * 60, color='g', linestyle='dotted', label='Breathing Frequency', linewidth=2)
axs[1, 2].annotate('Breathing BPM = %.2f' % (best_breathing_freq * 60), xy=(0.20, 0.90), xycoords='axes fraction', color='green', fontsize=10, weight='bold')
axs[1, 2].set_title('Breathing Region Filtered Data')
axs[1, 2].set_xlabel('Frequency (BPM)')
axs[1, 2].set_ylabel('Magnitude')

axs[1, 3].plot(fft_band_data_cardiac_freq * 60, np.abs(fft_band_data_cardiac))
axs[1, 3].axvline(x=best_cardiac_freq * 60, color='g', linestyle='dotted', label='Cardiac Frequency', linewidth=2)
axs[1, 3].annotate('Cardiac BPM = %.2f' % (best_cardiac_freq * 60), xy=(0.20, 0.90), xycoords='axes fraction', color='green', fontsize=10, weight='bold')
axs[1, 3].set_title('Cardiac Region Filtered Data')
axs[1, 3].set_xlabel('Frequency (BPM)')
axs[1, 3].set_ylabel('Magnitude')

if filename_truth_Br is not None and filename_truth_HR is not None:
    fig_GT, axs_GT = plt.subplots(2, 1, figsize=(20, 12))
    ground_truth_time = np.linspace(start_time, start_time + window_time, data_BR[start_time: start_time + window_time].shape[0])
    axs_GT[0].plot(ground_truth_time, data_BR[start_time: start_time + window_time])
    axs_GT[0].set_title('Breathing Rate')
    axs_GT[0].set_xlabel('Time')
    axs_GT[0].set_ylabel('Rate')
    axs_GT[0].legend(['Breathing Rate'])

    # Calculate the average Heart Rate
    average_Br = np.mean(data_BR[start_time: start_time + window_time])
    axs[1, 2].annotate('Ground truth BPM = %.2f' % average_Br, xy=(0.20, 0.80), xycoords='axes fraction', color='purple', fontsize=10, weight='bold')
    # Annotate the average Heart Rate
    axs_GT[0].annotate('Average BR = %.2f' % average_Br, xy=(0.30, 0.85), xycoords='axes fraction', color='green', fontsize=14, weight='bold')

    axs_GT[1].plot(ground_truth_time, data_HR[start_time: start_time + window_time])
    axs_GT[1].set_title('Heart Rate')
    axs_GT[1].set_xlabel('Time')
    axs_GT[1].set_ylabel('Rate')
    axs_GT[1].legend(['Heart Rate'])
    # Calculate the average Heart Rate
    average_HR = np.mean(data_HR[start_time: start_time + window_time])
    axs[1, 3].annotate('Ground truth BPM = %.2f' % average_HR, xy=(0.20, 0.80), xycoords='axes fraction', color='purple', fontsize=10, weight='bold')
    # Annotate the average Heart Rate
    axs_GT[1].annotate('Average HR = %.2f' % average_HR, xy=(0.30, 0.85), xycoords='axes fraction', color='red', fontsize=14, weight='bold')

plt.tight_layout()
plt.show()
print("done")
