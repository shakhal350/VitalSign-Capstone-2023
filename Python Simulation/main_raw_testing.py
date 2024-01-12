from data_processing import load_and_process_data
from plotting import setup_plots, create_animation

# Parameters and filename
filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_1.csv'
sample_window_size = 10240
update_interval = 1

# filename = r'C:\Users\Shaya\PycharmProjects\VitalSign-Capstone-2023\mmWave-VitalSign (Dataset Github)\RobustVSDataset_anonymous\p1\fix\2m\periodical\radar_csv\radar_01.csv'
# sample_window_size = 50
# update_interval = 1


# Load and process data
data, radar_parameters = load_and_process_data(filename)
print(radar_parameters["samplesPerFrame"], radar_parameters["frameRate"])

# print the number of data frames
print(f"Number of data frames: {len(data)}")

# Setup plots
fig, ax1, ax2, ax3, ax4, line1, line2, line3, line4 = setup_plots()

# Create and start animation
create_animation(fig, data, radar_parameters["samplesPerFrame"], radar_parameters["frameRate"], sample_window_size,
                 update_interval, line1,
                 line2,
                 line3, line4, ax1,
                 ax2,
                 ax3, ax4)
