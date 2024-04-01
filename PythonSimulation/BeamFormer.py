import numpy as np
import pandas as pd


def apply_weights(data):
    weights = [1, 1, 1, 1]
    tapering = np.hamming(4)
    weights = weights * tapering
    print("Weights: ", weights)
    # Apply the calculated weights to each receiver's data
    weighted_data = pd.DataFrame()
    for i in range(len(weights)):
        weighted_data[i] = data[i] * weights[i]
    return weighted_data
