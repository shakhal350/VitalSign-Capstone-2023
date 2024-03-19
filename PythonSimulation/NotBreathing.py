import numpy as np


def detect_non_breathing_periods(cleaned_chest_displacement, frameRate, window_size, threshold):
    window_samples = int(window_size * frameRate)
    non_breathing_periods = []
    start_time = None

    for i in range(0, len(cleaned_chest_displacement), window_samples):
        window = cleaned_chest_displacement[i:i + window_samples]
        if len(window) == 0: continue  # Skip empty windows at the end
        variance = np.var(window)
        displacement_range = np.max(window) - np.min(window)

        # Check if the window indicates non-breathing
        if variance < threshold and displacement_range < threshold:
            if start_time is None:  # Start of a new non-breathing period
                start_time = i / frameRate
        else:
            if start_time is not None:  # End of a non-breathing period
                end_time = i / frameRate
                non_breathing_periods.append((start_time, end_time))
                start_time = None

    # Check if there's an ongoing non-breathing period at the end of the data
    if start_time is not None:
        non_breathing_periods.append((start_time, len(cleaned_chest_displacement) / frameRate))

    return non_breathing_periods
