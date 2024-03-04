import pandas as pd


def load_and_process_data(filename):
    # For Children Dataset
    if 'Transposed' in filename:
        data = pd.read_csv(filename, header=None)
        for col in data.columns:
            data[col] = data[col].apply(lambda x: complex(x.replace('i', 'j')))
        data_avg = data[0] + data[1] + data[2] + data[3] / 4  # Average all 4 Rx Antennas
        return data_avg, get_radar_parameters("Children Dataset")

    # For VitalSign Dataset from Github
    if 'mmWave-VitalSign' in filename:
        data = pd.read_csv(filename, header=None)
        data = data[1]  # Only use one Rx Antenna (uses only Absolute Values)
        return data, get_radar_parameters("Github Dataset")

def load_and_process_data_NEW(filename,startInt, endInt):
    # For Children Dataset
    if 'Transposed' in filename:
        data = pd.read_csv(filename, header=None)
        for col in data.columns:
            data[col] = data[col].apply(lambda x: complex(x.replace('i', 'j')))
        data_avg = data[0] + data[1] + data[2] + data[3] / 4  # Average all 4 Rx Antennas
        return data_avg, get_radar_parameters("Children Dataset")

    # For VitalSign Dataset from Github
    if 'mmWave-VitalSign' in filename:
        data = pd.read_csv(filename, header=None)
        data = data[1]  # Only use one Rx Antenna (uses only Absolute Values)
        return data, get_radar_parameters("Github Dataset")


def get_radar_parameters(dataset_name):
    c = 3e8  # Speed of light in meters per second

    if dataset_name == "Children Dataset":
        r_para = {
            "rxNum": 4,  # Number of receivers
            "freqSlope": 40.8450012207031251e12,  # Frequency slope of the chirp in Hz/s
            "sampleRate": 3e6,  # Sample rate of the ADC in samples/s
            "bandwidth": 3.746303561822511e9,  # Bandwidth of the chirp in Hz
            "chirpLoops": 2,  # Number of loops chirped
            "adcSamples": 256,  # Number of ADC samples per chirp
            "startFreq": 60.25e9,  # Starting frequency of chirp in Hz
            "lambda": c / (3.746303561822511e9 / 2 + 60.25e9),  # Wavelength in meters
            "rangeResol": c / (2 * 3.746303561822511e9),  # Range resolution in meters
            "rangeMax": (3e6 * c) / (2 * 40.8450012207031251e12),  # Maximum range in meters
            "chirpTime": 3.746303561822511e9 / 40.8450012207031251e12,  # Chirp time in seconds
            "frameRate": 20,  # Frame rate in frames per second
            "samplesPerFrame": 512,  # Number of samples per frame
            "FFTSize": 2 ** 10,  # Size of FFT
            "rangeBin": (c * 3e6) / (2 * 40.8450012207031251e12 * 2 ** 10),  # Range bin in meters
        }
    elif dataset_name == "Github Dataset":
        r_para = {
            "rxNum": 1,  # Number of receivers
            "freqSlope": 68e12,  # Frequency slope of the chirp in Hz/s
            "sampleRate": 10.24e6,  # Sample rate of the ADC in samples/s
            "framePeriod": 0.1,  # Frame period in seconds
            "frameRate": 1.0 / 0.1,  # Frame rate in frames per second = 25 frames/sec
            "samplesPerFrame": 1,  # Number of samples per frame
            "startFreq": 77e9,  # Starting frequency of chirp in Hz
            "lambda": c / 77e9,  # Wavelength in meters
        }
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return r_para
