from scipy.signal import butter, lfilter, filtfilt


def lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


# Function to apply a Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def apply_high_pass_filter(data, cutoff_freq, fs, order=2):
    b, a = butter(order, cutoff_freq / (0.5 * fs), btype='high', analog=False)
    print(f"data before high pass filter: {data[:10]}")
    filtered_data = filtfilt(b, a, data)
    print(f"filtered_data after high pass filter: {filtered_data[:10]}")
    return filtered_data
