import numpy as np
from matplotlib import pyplot as plt

from SVD_processing import SVD_Matrix
from data_processing import load_and_process_data
from scipy.signal import butter, filtfilt, find_peaks

filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_13.csv'
data_Re, data_Im, radar_parameters = load_and_process_data(filename)

frameRate = radar_parameters['frameRate']
samplesPerFrame = radar_parameters['samplesPerFrame']

# pick a random number less than 290 to plot the data
random_number = np.random.randint(0, 290)
start = int(random_number * frameRate * samplesPerFrame)
end = int((random_number + 3) * frameRate * samplesPerFrame)

data_Re = data_Re[start:end]
data_Im = data_Im[start:end]

# SVD for noise reduction
data_Re = SVD_Matrix(data_Re, radar_parameters)
data_Im = SVD_Matrix(data_Im, radar_parameters)

adc_data_raw = data_Re + 1j * data_Im

# High-pass filter parameters
cutoff_freq = 0.05  # Cutoff frequency for the high-pass filter
order = 5  # Filter order

# Create high-pass filter (Butterworth)
b, a = butter(order, cutoff_freq / (0.5 * frameRate), btype='high', analog=False)
# Apply the filter to real and imaginary parts separately
filtered_real = filtfilt(b, a, np.real(adc_data_raw))
filtered_imag = filtfilt(b, a, np.imag(adc_data_raw))

# Recombine the filtered real and imaginary parts
filtered_data = filtered_real + 1j * filtered_imag

time_domain = np.linspace(random_number, random_number + 3, adc_data_raw.shape[0])

plt.figure()
plt.plot(time_domain, np.real(adc_data_raw))
plt.plot(time_domain, np.imag(adc_data_raw))
plt.title('Real Part of ADC Data')
# plt.show()

fft_data = np.fft.fft(adc_data_raw, axis=0)
fft_filtered_data = np.fft.fft(filtered_data, axis=0)
fft_freq = np.fft.fftfreq(adc_data_raw.shape[0], d=1 / frameRate)

# Keep only the positive half of the spectrum (including DC component)
half_index = len(fft_data) // 2
fft_data = fft_data[:half_index]
fft_filtered_data = fft_filtered_data[:half_index]
fft_freq = fft_freq[:half_index]

plt.figure()
plt.plot(fft_freq, np.abs(fft_data))
plt.plot(fft_freq, np.abs(fft_filtered_data), 'r')
plt.title('FFT of ADC Data')
# plt.show()

# Find peaks in the magnitude spectrum
peaks, properties = find_peaks(np.abs(fft_filtered_data), prominence=0.1, distance=1, width=1, height=np.max(np.abs(fft_filtered_data)) / 4)

# Take the median of the peaks detected (if peaks are found)
if peaks.size > 0:
    median_peak_freq = np.median(fft_freq[peaks])
    print(f'Median frequency of the peaks detected: {median_peak_freq} Hz')
else:
    print("No peaks detected.")

# Plot the FFT spectrum and the peaks
plt.figure()
plt.plot(fft_freq, np.abs(fft_filtered_data))
plt.plot(fft_freq[peaks], np.abs(fft_filtered_data)[peaks], "x")
plt.title('FFT of ADC Data with Peaks')
plt.show()

########################################################################################################################
# # Found the peaks in the frequency domain now need to get that regions phase differences to do phase unwrapping