import numpy as np
import pandas as pd


def complex_formatter(x):
    if isinstance(x, complex):
        return f"{x.real}+{x.imag}j" if x.imag >= 0 else f"{x.real}{x.imag}j"
    return x


def readDCA1000(fileName, csvFileName):
    # Configuration parameters
    numADCBits = 16  # number of ADC bits per sample
    numLanes = 4  # number of lanes is always 4
    isReal = 0  # set to 1 if real only data, 0 if complex data

    # Read .bin file
    with open(fileName, 'rb') as f:
        adcData = np.fromfile(f, dtype=np.int16)

    # Compensate for sign extension if ADC bits are not 16
    if numADCBits != 16:
        l_max = 2 ** (numADCBits - 1) - 1
        adcData[adcData > l_max] -= 2 ** numADCBits

    # Organize data by LVDS lane
    if isReal:
        # Reshape data based on one sample per LVDS lane for real only data
        adcData = np.reshape(adcData, (numLanes, -1))
    else:
        # Reshape and combine real and imaginary parts for complex data
        adcData = np.reshape(adcData, (-1, numLanes * 2))
        adcData = adcData[:, :numLanes] + 1j * adcData[:, numLanes:]

    # Convert to DataFrame and save to CSV
    # No need to transpose, as we've already arranged the array correctly
    df = pd.DataFrame(adcData)
    # Format each cell in DataFrame using the complex_formatter function
    formatted_df = df.map(complex_formatter)

    # Save the formatted DataFrame to CSV without index
    formatted_df.to_csv(csvFileName, index=False)
    print(f"Data saved to {csvFileName}")
    return adcData


# readDCA1000(r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\adc_data.bin", r"C:\Users\Shaya\OneDrive - Concordia University - Canada\UNIVERSITY\CAPSTONE\Our Datasets (DCA1000EVM)\CSVFiles(RawData)\ChestDisplacementData\March13th_Shayan_test\3\DCA1000EVM_Shayan_Shower_20Br_140Hr.csv")  # Enter file path