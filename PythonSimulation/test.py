import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c
from scipy.signal import find_peaks


from filter_processing import apply_high_pass_filter, butter_bandpass_filter, lowpass_filter
from SVD_processing import SVD_Matrix
from data_processing import load_and_process_data
from IPython.display import display, clear_output

def apply_windowing(time_signal):
    """
    Apply a window function to the time signal to reduce spectral leakage.
    """
    # Example: Hamming window
    window = np.hamming(len(time_signal))
    return time_signal * window


def compute_fft(time_signal, fRate):
    """
    Compute the FFT of the time signal, applying windowing to reduce spectral leakage.
    """
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


def find_significant_peaks(fft_data, fft_freq, prominence=0.1, width=1, percentile=99):
    percentile_threshold = np.percentile(np.abs(fft_data), percentile)
    peaks, properties = find_peaks(np.abs(fft_data), prominence=prominence, width=width, height=percentile_threshold)
    return peaks, properties


def select_best_peak(peaks, properties, fft_freq):
    if len(peaks) > 0:
        scores = properties['prominences'] * properties['widths'] * properties['peak_heights']
        best_peak_idx = np.argmax(scores)
        best_peak_freq = fft_freq[peaks[best_peak_idx]]
        print(f"Best Peak Frequency: {best_peak_freq} Hz")
    else:
        print("No peaks detected.")
        best_peak_freq = 0
    return best_peak_freq


def determine_frequency_window(best_peak_freq, fft_freq, window_gap=0.1):
    window_start_peak = max(best_peak_freq - window_gap, 0)
    window_end_peak = best_peak_freq + window_gap
    window_start_index = np.where(fft_freq > window_start_peak)[0][0]
    window_end_index = np.where(fft_freq > window_end_peak)[0][0]
    return window_start_peak, window_end_peak, window_start_index, window_end_index


# pick a random number for window time
window_time = 60  # <-----*** CHOOSE a window time YOU WANT ***
random_start = np.random.randint(60, 100)  # <-----*** Randomly picks a start time *** MAKE SURE TO CHANGE THIS TO A SPECIFIC NUMBER IF YOU WANT
random_start = 0
SVD_order = 4  # <-----*** CHOOSE a SVD order YOU WANT ***

sampleNumber = np.random.randint(1, 50)  # <-----*** Randomly picks a sample number *** MAKE SURE TO CHANGE THIS TO A SPECIFIC NUMBER IF YOU WANT TO TEST A SPECIFIC FILE
filename_truth_Br = None
filename_truth_HR = None
# filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_' + str(sampleNumber) + '.csv'
# filename_truth_Br = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Heart Rate & Breathing Rate\Breath_' + str(sampleNumber) + '.csv'
# filename_truth_HR = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Heart Rate & Breathing Rate\Heart_' + str(sampleNumber) + '.csv'
filename = r"C:\Users\Shaya\OneDrive - Concordia University - Canada\UNIVERSITY\CAPSTONE\Our Datasets (DCA1000EVM)\CSVFiles(RawData)\ChestDisplacementData\March13th_Shayan_test\1\DCA1000EVM__greynuns_12br_70hr.csv"
# filename = r"C:\Users\Shaya\OpenRadar\foo.csv"

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
print(f"Start: {start}", f"End: {end}", f"Fraction Frame Samples: {fraction_frame_samples}")
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
best_range_fft_peak_freq = select_best_peak(peaks_average, properties, fft_freq_average)

# Determine window start and end frequencies around the best peak
window_start_peak, window_end_peak, window_start_index, window_end_index = determine_frequency_window(best_range_fft_peak_freq, fft_freq_average)

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
    peaks, properties = find_peaks(np.abs(fft_frame), prominence=0.1, width=1, height=percentile_threshold)

    # Calculate a significance score for each peak
    scores = properties['prominences'] * properties['widths'] * properties['peak_heights']

    if len(scores) > 0:
        best_peak_idx = np.argmax(scores)
        best_range_fft_peak_freq = fft_freq_frame[peaks[best_peak_idx]]
    else:
        print("No peaks detected.")
        best_range_fft_peak_freq = 0

    # Find indices of the window around the best peak frequency
    # Ensure that window_start and window_end are within the available frequency range
    window_gap_added = 0  # Adjust as needed for your application
    window_start = max(best_range_fft_peak_freq - window_gap_added, fft_freq_frame[0])
    window_end = min(best_range_fft_peak_freq + window_gap_added, fft_freq_frame[-1])

    # Find the closest indices to window_start and window_end
    window_start_index = (np.abs(fft_freq_frame - window_start)).argmin()
    window_end_index = (np.abs(fft_freq_frame - window_end)).argmin()

    # Clear the current figure's content before plotting the next frame
    ax.clear()
    ax.plot(fft_freq_frame, np.abs(fft_frame))
    ax.axvline(x=window_start, color='k', linestyle='--', label='Window Start/End')
    ax.axvline(x=window_end, color='k', linestyle='--')
    ax.axvline(x=best_range_fft_peak_freq, color='g', linestyle='dotted', label='Center Peak', linewidth=2)
    ax.set_title('FFT of Filtered ADC Data')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude')
    ax.legend()
    clear_output(wait=True)
    display(fig_ani)
    plt.pause(0.0001)  # Pause briefly to allow the figure to be updated

    spectrogram_data.append(np.abs(fft_frame))  # Append the magnitude of the FFT result for the spectrogram
    phase_values_frame = np.angle(fft_frame[window_start_index:window_end_index + 1])
    phase_values.extend(phase_values_frame)  # Append to the overall phase values list

spectrogram_array = np.array(spectrogram_data)

# Plot the spectrogram using matplotlib
plt.figure(figsize=(10, 4))
plt.imshow(spectrogram_array.T, aspect='auto', origin='lower', extent=[0, len(spectrogram_data), fft_freq_frame[0], fft_freq_frame[-1]])
plt.colorbar(label='Magnitude')
plt.title('Spectrogram of All Chirps')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Chirp Number')
plt.show()

plt.close(fig_ani)
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
chest_displacement = ((lambda_c / (4 * np.pi)) * cleaned_diff_unwrap_phase) * 1000  # in mm

# grab name thats after the last backslash
name = filename.split("DCA1000EVM_")[-1]
# remove the .csv part
name, _ = os.path.splitext(name)

fft_chest_displacement, fft_phase_freq = compute_fft(chest_displacement, frameRate)

bandpass_chest_displacement_BR = butter_bandpass_filter(chest_displacement, 0.1, 0.8, frameRate, order=4)
bandpass_chest_displacement_HR = butter_bandpass_filter(chest_displacement, 0.8, 4.0, frameRate, order=4)

bandpass_phase_values_BR = butter_bandpass_filter(phase_values, 0.1, 0.8, frameRate, order=4)
bandpass_phase_values_HR = butter_bandpass_filter(phase_values, 0.8, 4.0, frameRate, order=4)

bandpass_unwrap_phase_BR = butter_bandpass_filter(unwrap_phase, 0.1, 0.8, frameRate, order=4)
bandpass_unwrap_phase_HR = butter_bandpass_filter(unwrap_phase, 0.8, 4.0, frameRate, order=4)

phase_values_fft, phase_freq_fft = compute_fft(bandpass_phase_values_BR, frameRate)
unwrap_phase_fft, unwrap_phase_freq = compute_fft(bandpass_phase_values_HR, frameRate)

fft_band_data_breathing, fft_band_data_breathing_freq = compute_fft(bandpass_chest_displacement_BR, frameRate)
fft_band_data_cardiac, fft_band_data_cardiac_freq = compute_fft(bandpass_chest_displacement_HR, frameRate)

best_breathing_freq_peaks, breathing_freq_peaks_properties = find_significant_peaks(fft_band_data_breathing, fft_band_data_breathing_freq, prominence=0.1, width=1, percentile=99)
best_cardiac_freq_peaks, cardiac_freq_peaks_properties = find_significant_peaks(fft_band_data_cardiac, fft_band_data_cardiac_freq, prominence=0.1, width=1, percentile=99)

# Select the peak with the highest significance score
best_breathing_freq = select_best_peak(best_breathing_freq_peaks, breathing_freq_peaks_properties, fft_band_data_breathing_freq)
best_cardiac_freq = select_best_peak(best_cardiac_freq_peaks, cardiac_freq_peaks_properties, fft_band_data_cardiac_freq)

print(f"Best Breathing Frequency: {best_breathing_freq * 60} BPM")
print(f"Best Cardiac Frequency: {best_cardiac_freq * 60} BPM")

# make a big subplot for all the plots I wrote
fig, axs = plt.subplots(4, 3, figsize=(20, 12))
fig.suptitle('All the plots for file: ' + filename)
axs[0, 0].plot(data_Re_small[:fraction_frame_samples], label='Real')
axs[0, 0].plot(data_Im_small[:fraction_frame_samples], label='Imaginary')
axs[0, 0].set_title('Part of Raw ADC Data Pre filtering')
axs[0, 0].set_xlabel('Time Domain')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].legend()

axs[0, 1].plot(data_Re_SVD[:fraction_frame_samples], label='Real')
axs[0, 1].plot(data_Im_SVD[:fraction_frame_samples], label='Imaginary')
axs[0, 1].set_title('Part of Filtered ADC Data')
axs[0, 1].set_xlabel('Time Domain')
axs[0, 1].set_ylabel('Amplitude')

axs[0, 2].plot(data_FFT_freq, data_fft)
axs[0, 2].set_title('FFT of ADC Data')
axs[0, 2].set_xlabel('Frequency')
axs[0, 2].set_ylabel('Magnitude')

axs[1, 0].axvline(x=window_start_peak, color='k', linestyle='--', label='Window Start/End')
axs[1, 0].axvline(x=window_end_peak, color='k', linestyle='--')
axs[1, 0].axvline(x=best_range_fft_peak_freq, color='g', linestyle='dotted', label='Center Peak', linewidth=2)
axs[1, 0].plot(fft_freq_average, np.abs(fft_filtered_data))
axs[1, 0].plot(fft_freq_average[peaks_average], np.abs(fft_filtered_data)[peaks_average], "o", label='peaks', color='r', markersize=2)
axs[1, 0].set_title('FFT of Filtered ADC Data')
axs[1, 0].set_xlabel('Frequency')
axs[1, 0].set_ylabel('Magnitude')
axs[1, 0].legend()

axs[1, 1].plot(phase_time, phase_values)
axs[1, 1].set_title('Phase Values')
axs[1, 1].set_xlabel('Time Domain')
axs[1, 1].set_ylabel('Phase')

axs[1, 2].plot(phase_time, unwrap_phase)
axs[1, 2].set_title('Unwrapped Phase Values')
axs[1, 2].set_xlabel('Time Domain')
axs[1, 2].set_ylabel('Phase')

axs[2, 0].plot(phase_time[:-1], chest_displacement)
axs[2, 0].set_title('Chest Displacement from Phase Differencing')
axs[2, 0].set_xlabel('Time')
axs[2, 0].set_ylabel('Chest Displacement (mm)')

axs[2, 1].plot(fft_phase_freq * 60, (np.abs(fft_chest_displacement)))
axs[2, 1].set_title('FFT of Chest Displacement')
axs[2, 1].set_xlabel('Frequency (BPM)')
axs[2, 1].set_ylabel('Magnitude')

axs[2, 2].plot(fft_band_data_breathing_freq * 60, np.abs(fft_band_data_breathing))
axs[2, 2].axvline(x=best_breathing_freq * 60, color='g', linestyle='dotted', label='Breathing Frequency', linewidth=2)
axs[2, 2].annotate('Best Breathing Frequency = %.2f' % (best_breathing_freq * 60), xy=(0.30, 0.85), xycoords='axes fraction', color='green', fontsize=14, weight='bold')
axs[2, 2].set_title('Breathing Region Filtered Data')
axs[2, 2].set_xlabel('Frequency (BPM)')
axs[2, 2].set_ylabel('Magnitude')

axs[3, 0].plot(fft_band_data_cardiac_freq * 60, np.abs(fft_band_data_cardiac))
axs[3, 0].axvline(x=best_cardiac_freq * 60, color='g', linestyle='dotted', label='Cardiac Frequency', linewidth=2)
axs[3, 0].annotate('Best Cardiac Frequency = %.2f' % (best_cardiac_freq * 60), xy=(0.30, 0.85), xycoords='axes fraction', color='green', fontsize=14, weight='bold')
axs[3, 0].set_title('Cardiac Region Filtered Data')
axs[3, 0].set_xlabel('Frequency (BPM)')
axs[3, 0].set_ylabel('Magnitude')

if filename_truth_Br is not None and filename_truth_HR is not None:
    ground_truth_time = np.linspace(random_start, random_start + window_time, data_BR[random_start: random_start + window_time].shape[0])
    axs[3, 1].plot(ground_truth_time, data_BR[random_start: random_start + window_time])
    axs[3, 1].set_title('Breathing Rate')
    axs[3, 1].set_xlabel('Time')
    axs[3, 1].set_ylabel('Rate')
    axs[3, 1].legend(['Breathing Rate'])
    # Calculate the average Heart Rate
    average_Br = np.mean(data_BR[random_start: random_start + window_time])
    # Annotate the average Heart Rate
    axs[3, 1].annotate('Average BR = %.2f' % average_Br, xy=(0.30, 0.85), xycoords='axes fraction', color='green', fontsize=14, weight='bold')

    axs[3, 2].plot(ground_truth_time, data_HR[random_start: random_start + window_time])
    axs[3, 2].set_title('Heart Rate')
    axs[3, 2].set_xlabel('Time')
    axs[3, 2].set_ylabel('Rate')
    axs[3, 2].legend(['Heart Rate'])
    # Calculate the average Heart Rate
    average_HR = np.mean(data_HR[random_start: random_start + window_time])
    # Annotate the average Heart Rate
    axs[3, 2].annotate('Average HR = %.2f' % average_HR, xy=(0.30, 0.85), xycoords='axes fraction', color='red', fontsize=14, weight='bold')

# phase_values_fft, phase_freq_fft = compute_fft(phase_values, frameRate)
# unwrap_phase_fft, unwrap_phase_freq = compute_fft(unwrap_phase, frameRate)
fig2, ax2 = plt.subplots(2, 2)
ax2[0, 0].plot(phase_freq_fft * 60, np.abs(phase_values_fft))
ax2[0, 0].set_title('FFT of Phase Values in Breathing Region')
ax2[0, 0].set_xlabel('BPM')
ax2[0, 0].set_ylabel('Magnitude')

ax2[0, 1].plot(unwrap_phase_freq * 60, np.abs(unwrap_phase_fft))
ax2[0, 1].set_title('FFT of Phase Values in Cardiac Region')
ax2[0, 1].set_xlabel('BPM')
ax2[0, 1].set_ylabel('Magnitude')

plt.tight_layout()
plt.show()

print("done")
