import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import inv


def mvdr_beamforming(IQ_data, steering_vector):
    """
    MVDR Beamforming implementation.

    :param IQ_data: Input I/Q data matrix of shape (number_of_receivers, number_of_samples)
    :param steering_vector: Steering vector for the desired direction, shape (number_of_receivers,)
    :return: Beamformed signal
    """
    # Ensure IQ_data is transposed if necessary
    if IQ_data.shape[0] > IQ_data.shape[1]:
        IQ_data = IQ_data.T
        print(f"IQ_data.shape (after transpose): ", IQ_data.shape)

    # Downsample the data
    downsample_copy = downsample_data(IQ_data, 1000)  # Transpose back after downsampling
    print(f"downsample_copy.shape: ", downsample_copy.shape)

    # Calculate covariance matrix of the received signals
    R = np.dot(downsample_copy, downsample_copy.conj().T) / downsample_copy.shape[1]

    # Calculate the inverse of the covariance matrix
    R_inv = inv(R)

    # Compute the weights for MVDR beamformer
    numerator = np.dot(R_inv, steering_vector)
    denominator = np.dot(steering_vector.conj().T, numerator)
    weights = numerator / denominator

    print(f"weights: ", weights)

    # Apply weights to the input data to get the beamformed signal
    beamformed_signal = np.dot(weights.conj().T, IQ_data)
    print(f"beamformed_signal.shape: ", beamformed_signal.shape)

    return beamformed_signal


def downsample_data(IQ_data, downsample_factor):
    """
    Downsample I/Q data by a specified factor.

    :param IQ_data: The original I/Q data array of shape (number_of_samples, number_of_receivers).
    :param downsample_factor: The factor by which to downsample the data.
    :return: Downsampled I/Q data.
    """
    # Select every Dth sample from the original data
    downsampled_IQ_data = IQ_data[:, ::downsample_factor]
    return downsampled_IQ_data
