import pandas as pd


def load_and_process_data(filename):
    data = pd.read_csv(filename, header=None)
    for col in data.columns:
        data[col] = data[col].apply(lambda x: complex(x.replace('i', 'j')))
    return data
