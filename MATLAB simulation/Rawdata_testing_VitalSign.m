%% Parameters Setting
clear
clc
close all

% Given parameters
AntNum = 4; % # of Rx Antennas
freqslope = 40.8450012207031251e12; % Frequency Slope in Hz/s
samplerate = 3000e3; % Sampling Rate in Hz
bw = 3.746303561822511e9; % Bandwidth in Hz
chirploops = 2; % # of Chirps in a Frame
adcsample = 256; % # of ADC Samples
startfreq = 60.25e9; % Start Frequency = Fc in Hz
c = 3e8; % Speed of Light in m/s
lambda = c/(bw/2 + startfreq); % Wavelength
rangeresol = c/(2*bw); % Range Resolution in meters
rangemax = (samplerate*c)/(2*freqslope); % Maximum Range in meters
tc = bw/freqslope; % Chirp Time in seconds
FFTSize = 2^10; % FFT Size for Range-FFT
RangeBin = (c * samplerate) / (2*freqslope*FFTSize); % Range Resolution of Range-FFT in meters
fps = 20; % Frames per Second
datalength = fps * 60 * 5; % # of Frames during Data Collection, assuming 5 minutes of data

%% Load the data
filename = 'C:\Users\omarm\Downloads/Transposed_Rawdata_1.csv'; % Replace with your file path
data = readmatrix(filename);
data = transpose(data);

% Assuming that the data is loaded correctly and the first 100,000 columns correspond to 100,000 samples
data_rx1 = data(1, 1:100000);
data_rx2 = data(2, 1:100000);
data_rx3 = data(3, 1:100000);
data_rx4 = data(4, 1:100000);

%% Process Data in Real-Time Simulation
% Initialize variables for range FFT
num_samples = adcsample * chirploops; % total samples per frame
frames_to_process = floor(length(data_rx1) / num_samples);

% Pre-allocate matrix for range FFT results
range_fft_results = zeros(FFTSize, frames_to_process, AntNum);

% Process each frame
for frame = 1:frames_to_process
    % Extract data for current frame
    idx_start = (frame - 1) * num_samples + 1;
    idx_end = frame * num_samples;
    
    % Perform Range-FFT for each antenna
    for rx = 1:AntNum
        % Extract samples for the current antenna and frame
        rx_data = data(rx, idx_start:idx_end);
        
        % Zero-padding to FFTSize
        rx_data_padded = [rx_data, zeros(1, FFTSize - num_samples)];
        
        % Perform the FFT
        range_fft = fft(rx_data_padded, FFTSize);
        
        % Store the result
        range_fft_results(:, frame, rx) = range_fft(1:FFTSize);
    end
end

%%
% Create figure for animation
h_fig = figure;
set(h_fig, 'Position', [100, 100, 600, 800]);

relevant_bins_start = 5;   % Start from bin 2 to skip the DC component at bin 1
relevant_bins_end = 500;   % Last bin of interest

% Number of frames to plot (can be less than frames_to_process for a shorter animation)
num_frames_to_plot = min(frames_to_process, datalength);

% Loop through each frame of data
for frame = 1:num_frames_to_plot
    % Initialize an array to accumulate the range profiles
    avg_range_profile = zeros(FFTSize, 1);
    
    % Sum up the range profiles from each receiver
    for rx = 1:AntNum
        % Select subplot for the current receiver
        subplot(AntNum+1, 1, rx); % Notice AntNum+1 to leave space for the average plot
        
        % Get the range FFT data for the current frame and receiver
        range_profile = abs(range_fft_results(:, frame, rx));
        
        % Accumulate the range profiles
        avg_range_profile = avg_range_profile + range_profile;
        
        % Plot the range profile for the current receiver
        plot(range_profile);
        xlim([1 FFTSize]); % Limit x-axis to FFT size
        ylim([0, max(range_profile)*1.1]); % Adjust y-axis dynamically
        title(['Receiver ', num2str(rx), ' - Frame ', num2str(frame)]);
        xlabel('Range Bin');
        ylabel('Magnitude');
        grid on;
    end

        % Calculate the average range profile
    avg_range_profile = avg_range_profile / AntNum;
    
    % Focus only on the relevant bins
    relevant_range_profile = avg_range_profile(relevant_bins_start:relevant_bins_end);
    
    % Plot the relevant part of the average range profile
    subplot(AntNum+1, 1, AntNum+1); % The last subplot is for the average profile
    plot(relevant_bins_start:relevant_bins_end, relevant_range_profile);
    xlim([relevant_bins_start relevant_bins_end]);
    title('Average Range Profile');
    xlabel('Range Bin');
    ylabel('Magnitude');
    grid on;
    
    % Find the peak within the relevant range bins
    [peak_value, peak_bin] = max(relevant_range_profile);
    
    % Correct the bin index based on the relevant range start
    peak_bin = peak_bin + (relevant_bins_start - 1);
    
    % Estimate the range of the target
    estimated_range = (peak_bin - 1) * RangeBin; % -1 because bin indexing starts at 1
    
    % Display the estimated range in meters
    disp(['Estimated Range for Frame ', num2str(frame), ': ', num2str(estimated_range), ' meters']);
    
    % Pause to control the speed of the animation
    pause(1/fps); % fps is the frames per second for the animation
    
    % Update the figure
    drawnow;
end
%%
% Assuming 'range_fft_results' is a matrix where each column is the FFT of a single chirp or frame
% and each row corresponds to a range bin.

% Isolate the FFT data for the 36th range bin across all frames
fft_36th_bin = range_fft_results(30:42, :);

% Convert from frequency domain to time domain using IFFT
signal_36th_bin = ifft(fft_36th_bin);

% Design the bandpass filter for breathing rate (0.1 to 0.5 Hz)
breathing_bpFilt = designfilt('bandpassiir', ...
               'FilterOrder', 4, ...
               'HalfPowerFrequency1', 0.1, ...
               'HalfPowerFrequency2', 0.5, ...
               'SampleRate', samplerate);

% Apply the bandpass filter
filtered_breathing_signal = filtfilt(breathing_bpFilt, signal_36th_bin);

% Plot the filtered signal to visualize breathing
figure;
plot((1:length(filtered_breathing_signal))/samplerate, filtered_breathing_signal);
xlabel('Time (s)');
ylabel('Amplitude');
title('Breathing Signal Isolated from 36th Range Bin');
grid on;