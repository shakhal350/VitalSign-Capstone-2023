import numpy as np
from matplotlib import pyplot as plt

from SVD_processing import SVD_Matrix
from data_processing import load_and_process_data

from scipy.fft import fft, fftfreq

filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_13.csv'
data_Re, data_Im, radar_parameters = load_and_process_data(filename)

# Apply SVD for noise reduction
data_Re = SVD_Matrix(data_Re, radar_parameters)
data_Im = SVD_Matrix(data_Im, radar_parameters)

# Calculate the wrapped phase
wrapped_phase = np.arctan2(data_Im, data_Re)

# Unwrap the phase to correct discontinuities
unwrapped_phase = np.unwrap(wrapped_phase, axis=0)  # Assuming the time axis is axis 0

# Plot the wrapped and unwrapped phase variations
time_vector = np.arange(unwrapped_phase.shape[0]) / radar_parameters['samplesPerSecond']

# Example values for demonstration (replace with actual values from radar_parameters)
c = 3e8  # Speed of light in meters per second
f = 77e9  # Frequency of radar signal in Hz, example for 77 GHz radar

# Calculate wavelength
lambda_wave = c / f

# Assuming unwrapped_phase from previous steps is available
# Calculate the difference in phase between successive measurements
# This represents delta_phi (Δφ)
delta_phi = np.diff(unwrapped_phase, axis=0)

# Calculate displacement delta_d (Δd) using the formula Δd = (λ / (4π)) * Δφ
# Note: The displacement calculation is based on the phase difference, so it's one less in length than the original phase array
delta_d = ((lambda_wave / (4 * np.pi)) * delta_phi) * 1000  # Convert to millimeters

# Update time vector for delta_d (one less in length)
time_vector_delta_d = time_vector[:-1]  # Removing the last time point

# Perform FFT on the displacement data
yf = fft(delta_d)
xf = fftfreq(len(delta_d), 1 / radar_parameters['frameRate'])

# Only consider the positive frequencies for analysis
positive_indices = xf > 0
xf_positive = xf[positive_indices]
yf_positive = np.abs(yf[positive_indices])

# Focus on the heart rate frequency range (0.5 Hz to 4 Hz) for typical human heart rates
heart_rate_freq_range = (xf_positive >= 0.1) & (xf_positive <= 4)
xf_heart_rate = xf_positive[heart_rate_freq_range]
yf_heart_rate = yf_positive[heart_rate_freq_range]

# Find the frequency with the maximum magnitude within the heart rate range
if len(xf_heart_rate) > 0:
    dominant_frequency_index = np.argmax(yf_heart_rate)
    dominant_frequency = xf_heart_rate[dominant_frequency_index]
    bpm = dominant_frequency * 60  # Convert Hz to BPM
else:
    bpm = 0  # In case no frequency is found within the range

print(f"Dominant Frequency: {dominant_frequency} Hz")
print(f"Estimated BPM: {bpm}")

# Calculate FFT of the raw Im data (assuming averaging over one of its dimensions)
# You might choose a specific signal from data_Im instead of averaging, depending on your needs
im_fft = fft(data_Im)
freqs = fftfreq(data_Im.shape[0], 1 / radar_parameters['frameRate'])

# Filter out the positive frequencies for plotting
positive_freqs = freqs > 0
im_fft_positive = im_fft[positive_freqs]
freqs_positive = freqs[positive_freqs]

# Set up the figure and subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # 3x2 grid of subplots

# Adjust layout for better spacing
plt.tight_layout(pad=4.0)

# Plot raw Im data at position [0, 0]
axs[0, 0].plot(time_vector, data_Im, label='Raw Im Data')  # Assuming mean for visualization
axs[0, 0].set_title('Raw Im Data Variation')
axs[0, 0].set_xlabel('Time (seconds)')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].legend()

# Plot FFT of raw Im data at position [0, 1]
axs[0, 1].plot(freqs_positive, np.abs(im_fft_positive), label='FFT of Raw Im Data')
axs[0, 1].set_title('FFT of Raw Im Data')
axs[0, 1].set_xlabel('Frequency (Hz)')
axs[0, 1].set_ylabel('Magnitude')
axs[0, 1].legend()

# Plot wrapped phase at position [1, 0]
axs[1, 0].plot(time_vector, wrapped_phase, label='Wrapped Phase')
axs[1, 0].set_title('Wrapped Phase Variation')
axs[1, 0].set_xlabel('Time (seconds)')
axs[1, 0].set_ylabel('Phase (radians)')
axs[1, 0].legend()

# Plot unwrapped phase at position [1, 1]
axs[1, 1].plot(time_vector, unwrapped_phase, label='Unwrapped Phase', color='orange')
axs[1, 1].set_title('Unwrapped Phase Variation')
axs[1, 1].set_xlabel('Time (seconds)')
axs[1, 1].set_ylabel('Phase (radians)')
axs[1, 1].legend()

# Plot change in displacement at position [2, 0]
axs[2, 0].plot(time_vector_delta_d, delta_d, label='Displacement Change', color='green')
axs[2, 0].set_title('Change in Displacement Over Time')
axs[2, 0].set_xlabel('Time (seconds)')
axs[2, 0].set_ylabel('Displacement (mm)')
axs[2, 0].legend()
axs[2, 0].grid(True)

# Plot FFT analysis with BPM at position [2, 1]
axs[2, 1].plot(xf_positive, np.abs(yf_positive), label='FFT of Displacement')
axs[2, 1].scatter(dominant_frequency, np.abs(yf_positive)[xf_positive == dominant_frequency], color='red', label=f'Dominant Frequency: {dominant_frequency:.2f} Hz\nBPM: {bpm:.0f}')
axs[2, 1].set_title('FFT Analysis & Estimated BPM')
axs[2, 1].set_xlim(0, 4)  # Focus on the heart rate frequency range
axs[2, 1].set_xlabel('Frequency (Hz)')
axs[2, 1].set_ylabel('Magnitude')
axs[2, 1].legend()

plt.show()