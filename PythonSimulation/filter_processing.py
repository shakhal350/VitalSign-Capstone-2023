from scipy.signal import butter, filtfilt, lfilter, iirnotch
import numpy as np


def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# Function to apply a Butterworth bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


def bandstop_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = filtfilt(b, a, data)
    return y

def gain_control_filter(data, threshold):
    newData = np.append(data, data[-1])
    x = np.reshape(newData, (-1, 20))
    e = np.sum(x*x,-1)
    
    for i in range(e.size):
        if e[i] > threshold:
            for j in range(x[i].size):
                x[i][j] = x[i][j] * np.sqrt(threshold / (e[i]))

    z = np.ravel(x)
    z = np.delete(z,-1)
    return z
