from data_processing import load_and_process_data
from plotting import setup_plots, create_animation

# Radar parameters
AntNum = 4  # Number of Rx Antennas
freqslope = 40.8450012207031251e12  # Frequency Slope in Hz/s
samplerate = 3e6  # Sampling Rate in Hz
bw = 3.746303561822511e9  # Bandwidth in Hz
chirploops = 2  # Number of Chirps in a Frame
adcsample = 256  # Number of ADC Samples
startfreq = 60.25e9  # Start Frequency = Fc in Hz
c = 3e8  # Speed of Light in m/s
lambda_ = c / (bw / 2 + startfreq)  # Wavelength
rangeresol = c / (2 * bw)  # Range Resolution in meters
rangemax = (samplerate * c) / (2 * freqslope)  # Maximum Range in meters
tc = bw / freqslope  # Chirp Time in seconds
fps = 20  # Frames per Second
samples_per_frame = 512

# Parameters and filename
filename = r'C:\Users\Shaya\Documents\MATLAB\CAPSTONE DATASET\CAPSTONE DATASET\Children Dataset\FMCW Radar\Rawdata\Transposed_Rawdata\Transposed_Rawdata_17.csv'
sample_window_size = 10240
cutoff_threshold = 1000
update_interval = 0.002

# Load and process data
data = load_and_process_data(filename)

data_avg = (data[0] + data[1] + data[2] + data[3]) / 4

# Setup plots
fig, ax1, ax2, ax3, ax4, line1, line2, line3, line4 = setup_plots()

# Create and start animation
create_animation(fig, data_avg, samples_per_frame, fps, sample_window_size, update_interval, cutoff_threshold, line1, line2,
                 line3, line4, ax1,
                 ax2,
                 ax3, ax4)
