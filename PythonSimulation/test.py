import numpy as np
from matplotlib import pyplot as plt

from SVD_processing import SVD_Matrix
from data_processing import load_and_process_data
from scipy.signal import butter, filtfilt, find_peaks

filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_4.csv'
# filename = r"C:\Users\Shaya\OneDrive - Concordia University - Canada\UNIVERSITY\CAPSTONE\Our Datasets (DCA1000EVM)\CSVFiles(RawData)\DCA1000EVM_shayan_fast_breathing.csv"
data_Re, data_Im, radar_parameters = load_and_process_data(filename)

frameRate = radar_parameters['frameRate']
samplesPerFrame = radar_parameters['samplesPerFrame']

# pick a random number less than 290 to plot the data
window_time = 120  # Time window to plot the data (in seconds)
random_number = np.random.randint(0, 40)
start = int(random_number * frameRate * samplesPerFrame)
end = int((random_number + window_time) * frameRate * samplesPerFrame)

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

time_domain = np.linspace(random_number, random_number + window_time, adc_data_raw.shape[0])

plt.figure()
plt.plot(time_domain, np.real(adc_data_raw))
plt.plot(time_domain, np.imag(adc_data_raw))
plt.title('Real Part of ADC Data')
plt.show()

fft_data = np.fft.fft(adc_data_raw, axis=0)
fft_filtered_data = np.fft.fft(filtered_data, axis=0)
fft_freq = np.fft.fftfreq(adc_data_raw.shape[0], d=1 / frameRate)

# Keep only the positive half of the spectrum
half_index = len(fft_data) // 2
fft_data = fft_data[:half_index]
fft_filtered_data = fft_filtered_data[:half_index]
fft_freq = fft_freq[:half_index]

plt.figure()
plt.plot(fft_freq, np.abs(fft_data))
plt.plot(fft_freq, np.abs(fft_filtered_data), 'r')
plt.title('FFT of ADC Data')
plt.show()

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
plt.plot(fft_freq[peaks], np.abs(fft_filtered_data)[peaks], "o", label='peaks', color='r')
plt.title('FFT of ADC Data with Peaks')
plt.show()

########################################################################################################################
# # Found the peaks in the frequency domain now need to get that regions phase differences to do phase unwrapping
phase_change = np.arctan2(np.imag(adc_data_raw), np.real(adc_data_raw))
phase_change_filtered = np.arctan2(np.imag(filtered_data), np.real(filtered_data))

# find the window in frequency domain of which the median_peak_freq is located and have a additional 0.5 Hz on each side
window_gap_added = 0.1
window_start = median_peak_freq - window_gap_added
window_end = median_peak_freq + window_gap_added
window_start_index = np.where(fft_freq > window_start)[0][0]
window_end_index = np.where(fft_freq > window_end)[0][0]

# plot the zoomed in FFT spectrum and the peaks
plt.figure()
plt.plot(fft_freq, np.abs(fft_filtered_data), 'r')
plt.plot(fft_freq[window_start_index:window_end_index], np.abs(fft_filtered_data)[window_start_index:window_end_index], 'g')
plt.title('Zoomed in FFT of ADC Data with Peaks')
plt.show()

signal_of_window = np.fft.ifft(fft_filtered_data[window_start_index:window_end_index], axis=0)

time_domain_window = np.linspace(random_number, random_number + window_time, signal_of_window.shape[0])
plt.figure()
plt.plot(time_domain_window, np.real(signal_of_window))
plt.plot(time_domain_window, np.imag(signal_of_window), 'r')
plt.title('Real Part and Im Part of signal_of_window')
plt.show()

phase_change_zoomed = np.arctan2(np.imag(signal_of_window), np.real(signal_of_window))

# Assuming phase_change_zoomed contains the wrapped phase information
unwrapped_phase = np.unwrap(phase_change_zoomed)

# Now unwrapped_phase can be used for further analysis
# Initialize the unwrapped phase array with the first value of the wrapped phase
unwrapped_phase_manual = np.zeros_like(phase_change_zoomed)
unwrapped_phase_manual[0] = phase_change_zoomed[0]

# Loop through the wrapped phase array and unwrap manually
for p in range(1, len(phase_change_zoomed)):
    if phase_change_zoomed[p] - unwrapped_phase_manual[p - 1] > np.pi:
        unwrapped_phase_manual[p] = phase_change_zoomed[p] - 2 * np.pi
    elif phase_change_zoomed[p] - unwrapped_phase_manual[p - 1] < -np.pi:
        unwrapped_phase_manual[p] = phase_change_zoomed[p] + 2 * np.pi
    else:
        unwrapped_phase_manual[p] = phase_change_zoomed[p]

# Ensure continuity by accumulating the adjustments (cumulative sum of differences)
unwrapped_phase_manual = np.cumsum(np.r_[unwrapped_phase_manual[0], np.diff(unwrapped_phase_manual)])

# Now you can plot the unwrapped phase
plt.figure()
plt.plot(time_domain_window, unwrapped_phase_manual, 'g')
plt.title('Manually Unwrapped Phase')
plt.show()

# Calculate the phase difference signal
phase_diff = np.diff(unwrapped_phase_manual)  # This is phi_dif(p)

# Apply smoothing to phase differences where jumps are detected
# This threshold can be adjusted based on the specifics of your data
threshold = np.pi  # A common threshold for phase jump detection

# Doing ϕinsert(p) = [ϕdiff(p − 1) + ϕdiff(p + 1)]/2
for p in range(1, len(phase_diff) - 1):  # Skip the first and last points for simplicity
    if abs(phase_diff[p]) > threshold:
        # Average the phase difference before and after the jump
        phase_diff[p] = (phase_diff[p - 1] + phase_diff[p + 1]) / 2

# Reconstruct the smoothed phase signal from phase differences
smoothed_phase = np.cumsum(np.insert(phase_diff, 0, unwrapped_phase_manual[0]))

# Now smoothed_phase contains your phase signal without jumps
# Plot the smoothed phase
plt.figure()
plt.plot(time_domain_window, smoothed_phase, 'g')  # time_domain_window should match the length of smoothed_phase
plt.title('Smoothed Phase Signal')
plt.show()
