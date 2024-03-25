import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from BeamFormer import mvdr_beamforming


def load_and_process_data(filename):
    # For Children Dataset
    if 'Transposed' in filename:
        data = pd.read_csv(filename, header=None)
        for col in data.columns:
            data[col] = data[col].apply(lambda x: complex(x.replace('i', 'j')))
        data = np.array(data)
        data = mvdr_beamforming(data, np.array([1, 2, 2, 1]))
        data_real = np.real(data)
        data_imag = np.imag(data)
        return data_real, data_imag, get_radar_parameters("Children Dataset")

    # For VitalSign Dataset from Github
    if 'DCA1000EVM' in filename:
        data = pd.read_csv(filename, header=None, skiprows=1)
        for col in data.columns:
            data[col] = data[col].apply(lambda x: complex(x))
        data = np.array(data)
        data = mvdr_beamforming(data, np.array([1, 2, 2, 1]))
        data_real = np.real(data)
        data_imag = np.imag(data)
        return data_real, data_imag, get_radar_parameters("DCA1000EVM")


def get_radar_parameters(dataset_name):
    c = 3e8  # Speed of light in meters per second

    if dataset_name == "Children Dataset":
        r_para = {
            "rxNum": 4,  # Number of receivers
            "freqSlope": 40.8450012207031251e12,  # Frequency slope of the chirp in Hz/s
            "sampleRate": 3e6,  # Sample rate of the ADC in samples/s (Fs)
            "bandwidth": 3.746303561822511e9,  # Bandwidth of the chirp in Hz (B)
            "chirpLoops": 2,  # Number of loops chirped
            "adcSamples": 256,  # Number of ADC samples per chirp
            "startFreq": 60.25e9,  # Starting frequency of chirp in Hz (fc)
            "lambda": c / (3.746303561822511e9 / 2 + 60.25e9),  # Wavelength in meters
            "rangeResol": c / (2 * 3.746303561822511e9),  # Range resolution in meters
            "rangeMax": (3e6 * c) / (2 * 40.8450012207031251e12),  # Maximum range in meters
            "chirpTime": 91.72e-6,  # Chirp Duration in seconds (Tc)
            "frameRate": 20,  # Frame rate in frames per second
            "samplesPerFrame": 2 * 256,  # Number of samples per frame
            "samplesPerSecond": 2 * 256 * 20,  # Number of samples per second
            "NumOfRangeBins": int(((3e6 * c) / (2 * 40.8450012207031251e12)) / ((c / (2 * 3.746303561822511e9))))  # Range bins
        }
    elif dataset_name == "DCA1000EVM":
        r_para = {
            "rxNum": 4,  # Number of receivers
            "freqSlope": 10.235e12,  # Frequency slope of the chirp in Hz/s
            "sampleRate": 3e6,  # Sample rate of the ADC in samples/s
            "bandwidth": 3.814e9,  # Bandwidth of the chirp in Hz
            "chirpLoops": 2,  # Number of loops chirped
            "adcSamples": 256,  # Number of ADC samples per chirp
            "startFreq": 77e9,  # Starting frequency of chirp in Hz
            "lambda": c / (3.814e9 / 2 + 77e9),  # Wavelength in meters -> lambda = c/(2B + fc)
            "rangeResol": c / (2 * 3.814e9),  # Range resolution in meters -> R_res = c/(2B)
            "rangeMax": 35.132,  # Maximum range in meters
            "chirpTime": 3.814e9 / 10.235e12,  # Chirp time in seconds -> T_chirp = B/slope
            "frameRate": 20,  # Frame rate in frames per second
            "samplesPerFrame": 2 * 256,  # Number of samples per frame
            "samplesPerSecond": 2 * 256 * 20,  # Number of samples per second
            "rangeBinInterval": 0.343,  # Range interval in meters
            "NumOfRangeBins": int(((3e6 * c) / (2 * 10.235e12)) / (c / (2 * 0.436907e9)))  # Range bins
        }

    return r_para
