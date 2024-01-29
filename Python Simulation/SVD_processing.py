import numpy as np


def SVD_Matrix(data_matrix, radar_parameters):
    """
    Apply Singular Value Decomposition (SVD) to the radar data matrix.

    Parameters:
    data_matrix (numpy.ndarray): A matrix representation of the radar data.

    Returns:
    U (numpy.ndarray): The left singular vectors.
    s (numpy.ndarray): The singular values.
    Vh (numpy.ndarray): The right singular vectors (conjugate transposed).
    """
    # Ensure radar_values is an array with the correct number of elements
    radar_values = np.array(data_matrix)  # data_matrix should be an array-like structure with radar values

    # Radar parameters (assuming they're passed correctly and include N)
    N = radar_parameters['samplesPerFrame']

    # Calculate the number of frames (K), ensuring it is an integer
    K = len(data_matrix) // N  # Use floor division to get an integer number of frames

    # Check if we need to truncate or pad the data
    total_samples = N * K
    if radar_values.size > total_samples:
        # Truncate the array to fit the shape (N, K)
        radar_values = radar_values[:total_samples]
    elif radar_values.size < total_samples:
        # Pad the array with zeros to fit the shape (N, K)
        radar_values = np.pad(radar_values, (0, total_samples - radar_values.size), 'constant')

    print(f"N: {N}, K: {K}")
    print(f"radar_values.size: {radar_values.size}")

    # Reshape the data into a matrix with N rows and K columns
    data_matrix = radar_values.reshape((N, K))

    # Perform SVD
    U, s, Vh = np.linalg.svd(data_matrix, full_matrices=False)

    print(f"U.shape: {U.shape}, s.shape: {s.shape}, Vh.shape: {Vh.shape}")
    noise_reduced_data = reduce_noise(U, s, Vh, 10)

    return noise_reduced_data


def reduce_noise(SVD_U, SVD_s, SVD_Vh, num_components):
    """
    Reconstruct the data matrix from the first 'num_components' singular values and vectors.

    Parameters:
    SVD_U, SVD_s, SVD_Vh: Outputs from the SVD_Matrix function.
    num_components: The number of singular values/vectors to use in the reconstruction.

    Returns:
    data_reduced: The reconstructed data matrix with reduced noise.
    """
    # Keep only the first 'num_components' singular values
    S = np.diag(SVD_s[:num_components])

    # Also keep only the corresponding columns of U and rows of Vh
    U_reduced = SVD_U[:, :num_components]
    Vh_reduced = SVD_Vh[:num_components, :]

    # Reconstruct the data matrix
    data_reduced = np.dot(U_reduced, np.dot(S, Vh_reduced))

    data_reduced_1D = data_reduced.flatten()

    return data_reduced_1D
