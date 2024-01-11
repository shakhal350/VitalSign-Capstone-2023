import pandas as pd


def load_and_process_data(filename):
    # For Children Dataset
    if 'Transposed' in filename:
        data = pd.read_csv(filename, header=None)
        for col in data.columns:
            data[col] = data[col].apply(lambda x: complex(x.replace('i', 'j')))
        data_avg = data[0] + data[1] + data[2] + data[3] / 4  # Average all 4 Rx Antennas
        return data_avg

    # For VitalSign Dataset from Github
    if 'mmWave-VitalSign' in filename:
        data = pd.read_csv(filename, header=None)
        data = data[1]  # Only use one Rx Antenna (uses only Absolute Values)
        return data
