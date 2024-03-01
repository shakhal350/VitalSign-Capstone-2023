import numpy as np
from matplotlib import pyplot as plt
from numpy import hanning
from scipy.ndimage import median_filter

from SVD_processing import SVD_Matrix
from data_processing import load_and_process_data
from scipy.signal import butter, filtfilt, find_peaks, lfilter

sampleNumber = 43

filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_' + str(sampleNumber) + '.csv'
filename_truth_Br = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Heart Rate & Breathing Rate\Breath_' + str(sampleNumber) + '.csv'
filename_truth_HR = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Heart Rate & Breathing Rate\Heart_' + str(sampleNumber) + '.csv'
# filename = r"C:\Users\Shaya\OneDrive - Concordia University - Canada\UNIVERSITY\CAPSTONE\Our Datasets (DCA1000EVM)\CSVFiles(RawData)\DCA1000EVM_shayan_31sec.csv"
data_Re, data_Im, radar_parameters = load_and_process_data(filename)

data_BR = np.genfromtxt(filename_truth_Br, delimiter=',')
data_HR = np.genfromtxt(filename_truth_HR, delimiter=',')

frameRate = radar_parameters['frameRate']
samplesPerFrame = radar_parameters['samplesPerFrame']

# pick a random number less than 290 to plot the data
window_time = 8  # Time window to plot the data (in seconds)
random_number = np.random.randint(20, 40)
start = int(random_number * frameRate * samplesPerFrame)
end = int((random_number + window_time) * frameRate * samplesPerFrame)

data_Re = data_Re[start:end]
data_Im = data_Im[start:end]

# SVD for noise reduction
data_Re = SVD_Matrix(data_Re, radar_parameters)
data_Im = SVD_Matrix(data_Im, radar_parameters)

# Apply the median filter to the real and imaginary parts separately
filtered_real_part = median_filter(data_Re, size=3)
filtered_imag_part = median_filter(data_Im, size=3)

# Recombine the filtered real and imaginary parts
adc_data_raw = filtered_real_part + 1j * filtered_imag_part

# Define the moving average window
window = np.ones(int(5)) / float(5)

# Apply the moving average filter to the real and imaginary parts separately
adc_data_raw_smoothed_real = np.convolve(np.real(adc_data_raw), window, 'same')
adc_data_raw_smoothed_imag = np.convolve(np.imag(adc_data_raw), window, 'same')

# Recombine the smoothed real and imaginary parts
adc_data_raw = adc_data_raw_smoothed_real + 1j * adc_data_raw_smoothed_imag

# Subtract the mean from both the real and imaginary parts
adc_data_raw = adc_data_raw - np.mean(adc_data_raw)

# High-pass filter parameters
cutoff_freq = 0.1  # Cutoff frequency for the high-pass filter
order = 4  # Filter order

# Create high-pass filter (Butterworth)
b, a = butter(order, cutoff_freq / (0.5 * frameRate), btype='high', analog=False)
# Apply the filter to real and imaginary parts separately
filtered_real = filtfilt(b, a, np.real(adc_data_raw))
filtered_imag = filtfilt(b, a, np.imag(adc_data_raw))

# Recombine the filtered real and imaginary parts
filtered_data = filtered_real + 1j * filtered_imag

time_domain = np.linspace(random_number, random_number + window_time, adc_data_raw.shape[0])

# Apply Hanning window to each frame
window = hanning(adc_data_raw.shape[0])
windowed_data = adc_data_raw * window

# Perform FFT on each frame
fft_data = np.fft.fft(windowed_data, axis=0)
fft_data = 20 * np.log10(np.abs(fft_data))

fft_data = fft_data - fft_data.mean(axis=0)

fft_filtered_data = np.fft.fft(filtered_data, axis=0)
fft_freq = np.fft.fftfreq(adc_data_raw.shape[0], d=1 / frameRate)

# Keep only the positive half of the spectrum
half_index = len(fft_data) // 2
fft_data = fft_data[:half_index]
fft_filtered_data = fft_filtered_data[:half_index]
fft_freq = fft_freq[:half_index]

percentile_threshold = np.percentile(np.abs(fft_filtered_data), 95)

# Find peaks with initial criteria
peaks, properties = find_peaks(np.abs(fft_filtered_data), prominence=0.1, width=1, height=percentile_threshold)

# Calculate a significance score for each peak (example formula, adjust as needed)
scores = properties['prominences'] * properties['widths'] * properties['peak_heights']

# Select the peak with the highest significance score
if len(scores) > 0:
    best_peak_idx = np.argmax(scores)
    best_peak_freq = fft_freq[peaks[best_peak_idx]]
    print(f"Best peak frequency: {best_peak_freq} Hz")
    print(f"Equals to", (3e8 * best_peak_freq) / (2 * radar_parameters["freqSlope"]), "m")
else:
    print("No peaks detected.")
    best_peak_freq = 0

########################################################################################################################
# # Found the peaks in the frequency domain now need to get that regions phase differences to do phase unwrapping

# find the window in frequency domain of which the median_peak_freq is located and have a additional 0.5 Hz on each side
window_gap_added = 0.1
if (best_peak_freq - window_gap_added) > 0:
    window_start = best_peak_freq - window_gap_added
else:
    window_start = 0
window_end = best_peak_freq + window_gap_added
window_start_index = np.where(fft_freq > window_start)[0][0]
window_end_index = np.where(fft_freq > window_end)[0][0]

signal_of_window = np.fft.ifft(fft_filtered_data[window_start_index:window_end_index], axis=0)

time_domain_window = np.linspace(random_number, random_number + window_time, signal_of_window.shape[0])

phase_change = np.arctan2(np.imag(signal_of_window), np.real(signal_of_window))
unwrapped_phase = np.unwrap(phase_change)
print(unwrapped_phase)
phase_diff = np.diff(unwrapped_phase)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


filtered_phase_diff = butter_bandpass_filter(phase_diff, 0.1, 3, frameRate, order=6)
lambda_c = 3e8 / 77e9

chest_displacement = ((lambda_c / (4 * np.pi)) * filtered_phase_diff) * 1000  # in mm

chest_displacement_fft = np.fft.fft(chest_displacement)
chest_displacement_fft_freq = np.fft.fftfreq(chest_displacement.shape[0], d=1 / frameRate)
half_index = len(chest_displacement_fft) // 2
chest_displacement_fft = chest_displacement_fft[:half_index]
chest_displacement_fft_freq = chest_displacement_fft_freq[:half_index]

# make a big subplot for all the plots i wrote
fig, axs = plt.subplots(4, 3, figsize=(15, 15))
fig.suptitle('All the plots')
axs[0, 0].plot(time_domain, np.real(adc_data_raw))
axs[0, 0].plot(time_domain, np.imag(adc_data_raw))
axs[0, 0].set_title('Real Part of ADC Data')

axs[0, 1].plot(fft_freq, np.abs(fft_data))
axs[0, 1].plot(fft_freq, np.abs(fft_filtered_data), 'r')
axs[0, 1].set_title('FFT of ADC Data')

axs[0, 2].plot(fft_freq, np.abs(fft_filtered_data))
axs[0, 2].plot(fft_freq[peaks], np.abs(fft_filtered_data)[peaks], "o", label='peaks', color='r')
axs[0, 2].set_title('FFT of ADC Data with Peaks')

axs[1, 0].plot(fft_freq, np.abs(fft_filtered_data), 'r')
axs[1, 0].plot(fft_freq[window_start_index:window_end_index], np.abs(fft_filtered_data)[window_start_index:window_end_index], 'g')
axs[1, 0].set_title('Zoomed in FFT of ADC Data with Peaks')

axs[1, 1].plot(time_domain_window, np.real(signal_of_window))
axs[1, 1].plot(time_domain_window, np.imag(signal_of_window), 'r')
axs[1, 1].set_title('Real Part and Im Part of signal_of_window')

axs[1, 2].plot(time_domain_window, unwrapped_phase, 'b')
axs[1, 2].set_title('unwrapped_phase')

axs[2, 0].plot(time_domain_window[:-1], phase_diff, 'r')
axs[2, 0].set_title('phase_diff')

axs[2, 1].plot(time_domain_window[:-1], chest_displacement, 'r')
axs[2, 1].set_title('Chest Displacement')

axs[2, 2].plot(chest_displacement_fft_freq * 60, np.abs(chest_displacement_fft))
axs[2, 2].set_title('FFT of Chest Displacement')

axs[3, 0].plot(data_BR[random_number:random_number + window_time])
axs[3, 0].set_title('Breathing Rate')

axs[3, 1].plot(data_HR[random_number:random_number + window_time])
axs[3, 1].set_title('Heart Rate')

plt.tight_layout()
plt.show()
