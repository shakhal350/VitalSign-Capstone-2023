import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import numpy as np


class RadarPhaseDataset(Dataset):
    def __init__(self, features, labels=None, train=True):
        self.features = features
        self.labels = labels
        self.train = train


    def __len__(self):
        return len(self.features)


    def __getitem__(self, idx):
        if self.train or self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

def segment_data(phase_data, heart_rates, window_size, sampling_rate):

    num_samples_per_window = int(window_size * sampling_rate)
    segmented_phase_data = []
    segmented_heart_rates = []

    # Split phase data and heart rates into windows
    for i in range(0, len(phase_data), num_samples_per_window):
        if i + num_samples_per_window <= len(phase_data):  # Ensure no index out of bounds
            segmented_phase_data.append(phase_data[i:i + num_samples_per_window])
            window_heart_rates = heart_rates[i:i + num_samples_per_window]
            segmented_heart_rates.append(np.mean(window_heart_rates))  # Or np.median, or window_heart_rates[-1]

    return np.array(segmented_phase_data), np.array(segmented_heart_rates)



def load_and_preprocess_single_file(data_path, window_size=1, sampling_rate=20):
    df = pd.read_csv(data_path)

    phase_data = df.iloc[:, 1].values  # 2nd column (phase values)
    heart_rates = df.iloc[:, 2].values  # 3rd column (heart rate values)

    # Scale the phase data 
    scaler = StandardScaler()
    phase_data_scaled = scaler.fit_transform(phase_data.reshape(-1, 1))  # Reshape needed for single feature
    
    # Segment the data into windows
    segmented_phase_data, segmented_heart_rates = segment_data(phase_data_scaled, heart_rates, window_size, sampling_rate)
    return segmented_phase_data, segmented_heart_rates


# to handle multiple files
def load_and_preprocess_multiple_files(file_paths, window_size=1, sampling_rate=20):
    all_segmented_phase_data = []
    all_segmented_heart_rates = []
    
    for file_path in file_paths:
        segmented_phase_data, segmented_heart_rates = load_and_preprocess_single_file(file_path, window_size, sampling_rate)
        all_segmented_phase_data.append(segmented_phase_data)
        all_segmented_heart_rates.append(segmented_heart_rates)
   
    all_segmented_phase_data = [arr for arr in all_segmented_phase_data if arr.size > 0]

    for i, arr in enumerate(all_segmented_phase_data):
        if arr.ndim == 1:  # It's a flat array; likely only one segment
            all_segmented_phase_data[i] = arr.reshape(1, -1)  # Reshape to 2D
        elif arr.ndim == 2:  # It's a 2D array but should be 3D
            all_segmented_phase_data[i] = arr.reshape(1, *arr.shape)  # Add the segments dimension

    # Combine data from all files
    all_segmented_phase_data = np.concatenate(all_segmented_phase_data, axis=0)
    all_segmented_heart_rates = np.concatenate(all_segmented_heart_rates, axis=0)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(all_segmented_phase_data, all_segmented_heart_rates, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def create_datasets(X_train, X_test, y_train, y_test):
    train_dataset = RadarPhaseDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = RadarPhaseDataset(torch.FloatTensor(
        X_test), torch.FloatTensor(y_test), train=False)
    return train_dataset, test_dataset



