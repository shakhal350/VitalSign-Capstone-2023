import numpy as np

from SVD_processing import SVD_Matrix, reduce_noise
from data_processing import load_and_process_data
from plotting import create_animation

# Parameters and filename
filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_11.csv'
sample_window_size = 40960
update_interval = 2

# filename = r'C:\Users\Shaya\PycharmProjects\VitalSign-Capstone-2023\mmWave-VitalSign (Dataset Github)\RobustVSDataset_anonymous\p1\fix\2m\periodical\radar_csv\radar_01.csv'
# sample_window_size = 50
# update_interval = 0.1


# Load and process data
data, radar_parameters = load_and_process_data(filename)
print(radar_parameters["samplesPerFrame"], radar_parameters["frameRate"])

SVD_U, SVD_s, SVD_Vh = SVD_Matrix(np.abs(data), radar_parameters)
data = reduce_noise(SVD_U, SVD_s, SVD_Vh, num_components=5)

# Create and start animation
create_animation(data, radar_parameters["samplesPerFrame"], radar_parameters["frameRate"], sample_window_size,
                 update_interval,
                 )
