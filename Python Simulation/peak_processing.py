from scipy.signal import find_peaks


def find_significant_peaks(fft_magnitude, min_distance=2, min_height=None, min_prominence=None):
    peaks, properties = find_peaks(fft_magnitude, distance=min_distance, height=min_height, prominence=min_prominence)
    peak_heights = properties["peak_heights"]

    # Sort peaks based on peak heights in descending order
    ordered_peaks = sorted(zip(peaks, peak_heights), key=lambda x: x[1], reverse=True)

    # You can then use this ordered list to find the best general area for the highest peak
    # or use the `n` highest peaks for your analysis.

    # Check if ordered_peaks is empty
    if not ordered_peaks:
        print("No significant peaks found.")
        return None  # or return a default value

    return ordered_peaks[0][0]
