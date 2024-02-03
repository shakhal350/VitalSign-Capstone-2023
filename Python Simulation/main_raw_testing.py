from SVD_processing import SVD_Matrix
from data_processing import load_and_process_data
from plotting import create_animation

# Parameters and filename
# filename = r'C:\Users\Shaya\Downloads\DCA1000EVM_shayan.csv'
# filename = r"C:\Users\Shaya\OneDrive - Concordia University - Canada\UNIVERSITY\CAPSTONE\Our Datasets (DCA1000EVM)\CSVFiles(RawData)\DCA1000EVM_shayan_normal_breathing.csv"
filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_11.csv'
# filename = r"C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Walking AWR16x\Walking_adc_DataTable.csv"

# Load and process data
data, radar_parameters = load_and_process_data(filename)

animation_update_interval = 1

data = SVD_Matrix(data, radar_parameters)

print(data)

# Create and start animation
create_animation(data, radar_parameters, animation_update_interval, timeWindowMultiplier=5)
