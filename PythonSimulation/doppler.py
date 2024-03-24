import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

def perform_doppler_fft(data, samples_per_chirp, num_chirps, Tc, lambda_c):
    """
    Perform Doppler FFT on the radar data and estimate velocity.

    Parameters:
    - data: 2D array of radar data after Range FFT, dimensions: [num_chirps, samples_per_chirp]
    - samples_per_chirp: Number of samples per chirp.
    - num_chirps: Number of chirps per frame.
    - Tc: Time between 2 consecutive chirps.
    - lambda_c: Wavelength of the transmit signal.

    Returns:
    - velocity_estimates: Velocity estimates for each range bin.
    - doppler_shifted: The Doppler-shifted signal (2D FFT result).
    """
    # Perform Doppler FFT along the chirps dimension
    doppler_fft = fft(data, axis=0)
    doppler_fft_shifted = fftshift(doppler_fft, axes=0)

    # Calculate frequency bins
    doppler_frequency_bins = np.linspace(-num_chirps / (2*Tc), num_chirps / (2*Tc), num_chirps)

    # Calculate phase difference from the shifted FFT results
    phase_difference = np.angle(doppler_fft_shifted)

    # Velocity estimation from phase difference
    # v = lambda * omega / (2 * pi * Tc)
    # Omega is the frequency corresponding to the peak phase difference for each range bin
    omega = 2 * np.pi * doppler_frequency_bins[np.argmax(np.abs(doppler_fft_shifted), axis=0)]
    velocity_estimates = lambda_c * omega / (2 * np.pi * Tc)

    return velocity_estimates, doppler_fft_shifted


def doppler_fft_spectrogram(data, frameRate, samplesPerFrame, window='hamming'):
    """
    Perform frame-by-frame Doppler FFT analysis and plot the spectrogram.

    Parameters:
    - data: Complex-valued input signal (data_Re + 1j*data_Im) from the radar.
    - frameRate: The rate at which frames are captured.
    - samplesPerFrame: Number of samples per frame.
    - window (optional): Type of windowing function to apply. Defaults to 'hamming'.

    Note: This function assumes 'data' is a 1D numpy array of complex values.
    """
    # Determine the number of frames
    global fft_result
    num_frames = int(np.floor(len(data) / samplesPerFrame))

    # Pre-allocate the spectrogram matrix
    spectrogram = np.zeros((samplesPerFrame // 2, num_frames))

    # Define window function
    if window == 'hamming':
        win = np.hamming(samplesPerFrame)
    else:
        # You can add more window functions as needed
        win = np.ones(samplesPerFrame)

    # Loop over each frame
    for i in range(num_frames):
        # Extract the current frame with windowing
        start_idx = i * samplesPerFrame
        end_idx = start_idx + samplesPerFrame
        frame = data[start_idx:end_idx] * win

        # Perform FFT on the frame
        fft_result = fft(frame)

        # Keep only the positive frequencies
        half = len(fft_result) // 2
        fft_result = fft_result[:half]

        # Calculate magnitude spectrum and store in the spectrogram matrix
        spectrogram[:, i] = np.abs(fft_result)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower',
               extent=[0, num_frames / frameRate, 0, frameRate / 2])
    plt.colorbar(label='Magnitude')
    plt.title('Doppler FFT Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')


    # Parameters
    Tc = 0.050  # Time between chirps, for example, 1 ms
    lambda_c = 3e8 / 77e9  # Wavelength of the radar signal, assuming a 77 GHz radar

    # Example data dimensions (to be replaced with actual data dimensions)
    samples_per_chirp = 256  # Number of samples per chirp, adjust based on your data
    num_chirps = 1200  # Number of chirps per frame, adjust based on your data

    # Generate example data (replace this with your actual radar data after Range FFT)
    data_after_range_fft = spectrogram

    # Perform Doppler FFT and velocity estimation
    velocity_estimates, doppler_fft_result = perform_doppler_fft(data_after_range_fft, samples_per_chirp, num_chirps, Tc, lambda_c)
    print(spectrogram.shape)
    # Plotting the Doppler FFT result (spectrogram)
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(doppler_fft_result), aspect='auto', origin='lower',
               extent=[0, samples_per_chirp, -num_chirps / (2 * Tc), num_chirps / (2 * Tc)])
    plt.colorbar(label='Magnitude')
    plt.title('Doppler FFT Spectrogram (Velocity Estimation)')
    plt.ylabel('Doppler Frequency (Hz)')
    plt.xlabel('Range Bin')
    plt.show()



