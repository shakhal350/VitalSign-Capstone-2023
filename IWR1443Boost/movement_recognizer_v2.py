import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import statistics
from tkinter import *
import tkinter as tk
#Libreria de hilos
from threading import Thread

# import warnings
# warnings.filterwarnings("ignore")

# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     # warnings.filterwarnings("ignore")
#     fxn()

# Change the configuration file name
# configFileName = './scripts_python/IWR1443-Read-Data-Python-MMWAVE-SDK-1-master/IWR1443-Read-Data-Python-MMWAVE-SDK-1-master/1443config.cfg'
configFileName = 'profile.cfg'

CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0

# - Global variables for movement recognition -----------------------------------
mean_y_prev_list = [0]
mean_y_prev = np.mean(mean_y_prev_list)
mean_x_prev_list = [0]
mean_x_prev = np.mean(mean_x_prev_list)
decission_y = []
decission_x = []
final_decission_y_old = ""
final_decission_x_old = ""

# --------------------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    """
    Configures the serial ports and sends the CLI commands to the radar. 
    It outputs the serial objects for the data and CLI ports.
    """
    
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports
    
    # Raspberry pi
    #CLIport = serial.Serial('/dev/ttyACM0', 115200)
    #Dataport = serial.Serial('/dev/ttyACM1', 921600)
    
    # Windows
    CLIport = serial.Serial('COM4', 115200)
    Dataport = serial.Serial('COM6', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)
        
    return CLIport, Dataport
# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    """
     Parses the configuration file to extract the configuration parameters. 
     It returns the configParameters dictionary with the extracted parameters.
    """
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
                
            digOutSampleRate = int(splitWords[11])
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            
            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = int(splitWords[5]);

            
    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    
    return configParameters # returns dictionary containing config key values
# ------------------------------------------------------------------

# Funtion to read and parse the incoming data
def readAndParseData14xx(Dataport, configParameters):
    """
    It reads the data from the data serial port and parses the recived buffer to extract the data 
    from the Detected objects package only. Othe package types (range profile, range-azimuth heat map...) 
    could be done similarly but have not been implemented yet. This functions returns a boolean variable 
    (dataOK) that stores if the data has been correctly, the frame number and the detObj dictionary with 
    the number of detected objects, range (m), doppler velocity (m/s), peak value and 3D position (m).
    """
    global byteBuffer, byteBufferLength
    
    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2**15
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    MMWDEMO_UART_MSG_RANGE_PROFILE   = 2
    maxBufferSize = 2**15
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
    
    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}
    
    readBuffer = Dataport.read(Dataport.in_waiting) # read buffer into readBuffer from serial data port
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8') # store read data from buffer in var byteVec
    byteCount = len(byteVec)
    
    # byteBufferLength = 0
    # Check that the buffer is not full, and then add the data to the buffer and update byteBufferLength if requiered
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount
        
    # Check that the buffer has some data
    if byteBufferLength > 16: # apparently it has to have at least 16 bytes in buffer
        
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0] # It returns a tuple of indices if an only condition is given, the indices where the condition is True.

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+8]
            if np.all(check == magicWord):
                startIdx.append(loc)
               
        # Check that startIdx is not empty
        if startIdx:
            
            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
                byteBufferLength = byteBufferLength - startIdx[0]
                
            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
                
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            
            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12+4],word) # Matrix product of two arrays
            # total packet lenght stored in 12:12+4 position in 4 bytes in the packet
            
            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1
    
    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2**8, 2**16, 2**24]
        
        # Initialize the pointer index
        idX = 0
        
        # Read the header
        magicNumber = byteBuffer[idX:idX+8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX+4],word) # number of Data Structures in package (4 bytes)
        idX += 4
        
        # UNCOMMENT IN CASE OF SDK 2
        #subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        #idX += 4
        
        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX:idX+4],word) # structure tag
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX:idX+4],word) # length of structure
            idX += 4
            
            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
            
                # word array to convert 2 bytes to a 16 bit number
                word = [1, 2**8]
                tlv_numObj = np.matmul(byteBuffer[idX:idX+2],word) 
                idX += 2 # ------------???
                tlv_xyzQFormat = 2**np.matmul(byteBuffer[idX:idX+2],word) # shouldnt it be 4 bytes after, not 2?
                idX += 2
                
                # Initialize the arrays
                rangeIdx = np.zeros(tlv_numObj,dtype = 'int16') # type int16 because they are 2 bytes values
                dopplerIdx = np.zeros(tlv_numObj,dtype = 'int16')
                peakVal = np.zeros(tlv_numObj,dtype = 'int16')
                x = np.zeros(tlv_numObj,dtype = 'int16')
                y = np.zeros(tlv_numObj,dtype = 'int16')
                z = np.zeros(tlv_numObj,dtype = 'int16')
                
                for objectNum in range(tlv_numObj):
                    
                    # Read the data for each object
                    rangeIdx[objectNum] =  np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    dopplerIdx[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    peakVal[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    x[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    y[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    z[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    
                # Make the necessary corrections and calculate the rest of the data
                rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"]/2 - 1)] = dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"]/2 - 1)] - 65535
                dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                #x[x > 32767] = x[x > 32767] - 65536
                #y[y > 32767] = y[y > 32767] - 65536
                #z[z > 32767] = z[z > 32767] - 65536
                x = x / tlv_xyzQFormat
                y = y / tlv_xyzQFormat
                z = z / tlv_xyzQFormat
                
                # Store the data in the detObj dictionary
                detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                          "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}
                
                dataOK = 1             
        
  
        # Remove already processed data
        if idX > 0 and byteBufferLength > idX:
            shiftSize = totalPacketLen
               
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8')
            byteBufferLength = byteBufferLength - shiftSize
            
            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
                

    return dataOK, frameNumber, detObj
# ------------------------------------------------------------------

# Funtion to update the data and display in the plot
def update():
    dataOk = 0
    global detObj, s, p, ptr, mean_y_prev_list, mean_y_prev, decission_y, decission_x, mean_x_prev_list, mean_x_prev, configParameters
    global final_decission_y_old, final_decission_x_old
    x = []
    y = []
    peak_val = []
    range = []
    y_ok = []
    x_ok = []
    
    min_y = 0.1 # m 
    max_y = 2 # m

    min_x = -1.5 # m 
    max_x = 1.5 # m

    delta_y = 0.01 # m
    delta_x = 0.02
    # Read and parse the received data
    dataOk, frameNumber, detObj = readAndParseData14xx(Dataport, configParameters)
    
    # print(detObj)

    if dataOk and len(detObj["x"]) > 0:
        #print(detObj)
        x = detObj["x"]
        y = detObj["y"]
        # peak_val = detObj["peakVal"] # idk if those are dBm or what
        # range = detObj["range"]
        # v_doppler = detObj["doppler"]
        # numObjts = detObj["numObj"]
       
        i = 0
        for _ in y:
            # print(i)
            if (y[i]<max_y and y[i]>min_y) and (x[i]<max_x and x[i]>min_x):
                y_ok.append(y[i])
                x_ok.append(x[i])
            i = i+1

        if (len(y_ok)>0) and (len(x_ok)>0):
            mean_y = np.mean(y_ok)
            mean_x = np.mean(x_ok)
        
            if abs(mean_y-mean_y_prev) < delta_y:
                decission_y.append('y_estatico')
            elif mean_y > mean_y_prev:
                decission_y.append('y_alejarse')
            elif mean_y < mean_y_prev:
                decission_y.append('y_acercarse')

            if abs(mean_x-mean_x_prev) < delta_x:
                decission_x.append('x_estatico')
            elif mean_x > mean_x_prev:
                decission_x.append('x_derecha')
            elif mean_x < mean_x_prev:
                decission_x.append('x_izquierda')  
            
            
            if len(decission_y)>10 and len(decission_x)>10:
                final_decission_y = statistics.mode(decission_y)
                final_decission_x = statistics.mode(decission_x)

                decission_y = []
                decission_x = []

                if final_decission_y == "y_estatico":
                    if final_decission_x == "x_estatico":
                        Mov_Dec_label.config(text = "GESTO 0: Objetivo estático", fg="white")
                        print("GESTO 0: Objetivo estático")
                    elif final_decission_x == "x_derecha":
                        Mov_Dec_label.config(text = "GESTO 1: Objetivo moviendose hacia la derecha", fg="white")
                        print("GESTO 1: Objetivo moviendose hacia la derecha")
                    elif final_decission_x == "x_izquierda":
                        Mov_Dec_label.config(text = "GESTO 2: Objetivo moviendose hacia la izquierda", fg="white")
                        print("GESTO 2: Objetivo moviendose hacia la izquierda")

                elif final_decission_y == "y_alejarse":
                    if final_decission_x == "x_estatico":
                        Mov_Dec_label.config(text = "GESTO 3: Objetivo alejandose", fg="white")
                        print("GESTO 3: Objetivo alejandose")
                    elif final_decission_x == "x_derecha":
                        Mov_Dec_label.config(text = "GESTO 4: Objetivo alejandose hacia la derecha", fg="white")
                        print("GESTO 4: Objetivo alejandose hacia la derecha")
                    elif final_decission_x == "x_izquierda":
                        Mov_Dec_label.config(text = "GESTO 5: Objetivo alejandose hacia la izquierda", fg="white")
                        print("GESTO 5: Objetivo alejandose hacia la izquierda")

                elif final_decission_y == "y_acercarse":
                    if final_decission_x == "x_estatico":
                        Mov_Dec_label.config(text = "GESTO 6: Objetivo acercandose", fg="white")
                        print("GESTO 6: Objetivo acercandose")
                    elif final_decission_x == "x_derecha":
                        Mov_Dec_label.config(text = "GESTO 7: Objetivo acercandose hacia la derecha", fg="white")
                        print("GESTO 7: Objetivo acercandose hacia la derecha")
                    elif final_decission_x == "x_izquierda":
                        Mov_Dec_label.config(text = "GESTO 8: Objetivo acercandose hacia la izquierda", fg="white")
                        print("GESTO 8: Objetivo acercandose hacia la izquierda")

            mean_y_prev = mean_y
            mean_x_prev = mean_x

            x = [mean_x]
            y = [mean_y]

            s.setData(x, y)
            
            # Vamos a descartar en primer lugar todos los objects que esten fuera del rango preestablecido y hacer una media del rango de los restantes
            # esa media del rango de los restantes sera comparada con la media obtenida en la medida anterior para comparar y saber si el
            # movimiento es en una direccion u otra
        else:
            Mov_Dec_label.config(text = '\n\nNo object identified within range\n\n', fg="white")
            print('\n\nNo object identified within range\n\n')

    return dataOk
# ------------------------------------------------------------------

# --------------------[ QTGraph App ]-------------------------------
# START QtAPP for the plot
app = pg.mkQApp("Scattering Plot")

win = pg.GraphicsLayoutWidget(show=True, title="Radar")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')
pg.setConfigOptions(antialias=True)

# Set the plot ----------------------------
pg.setConfigOption('background','w')

p = win.addPlot(title="Detected Objects: XY")
p.setXRange(-1.5,1.5)
p.setYRange(0,2)
p.setLabel('left',text = 'Y position (m)')
p.setLabel('bottom', text= 'X position (m)')
p.showGrid(x=True, y=True, alpha=True)
s = p.plot([],[],pen=None,symbol='x')

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

# pg.exec()


# -------------------- [ TK-GUI] ---------------------
window = tk.Tk()
window.geometry("1200x400") # Width x Height
window.title("Movemente Recognizer Radar")
window.resizable(True, True)
window.configure(bg="black")

#TITULO de MENSAJE de movement decission
title_Mov_Dec = Label(window,text="Movement decission:", anchor = "w", bg="black",fg="white",font=("Helvetica", 30, 'bold'))
title_Mov_Dec.place(x=300,y=150, width= 400, height=40)
#LABEL de MENSAJE de movement decission
Mov_Dec_label = Label(window,text="",anchor = "w", bg="black",fg="white",font=("Helvetica", 30, 'bold'))
Mov_Dec_label.place(x=80,y=300, width= 1100, height=40)

def run():
    global configParameters, ptr
    # Configurate the serial port
    CLIport, Dataport = serialConfig(configFileName)

    # Get the configuration parameters from the configuration file
    configParameters = parseConfigFile(configFileName)

    ptr = 0

    detObj = {}  
    frameData = {}    
    currentIndex = 0

    # pg.exec()

    while True:
        try:
            # Update the data and check if the data is okay
            dataOk = update()

            if dataOk:
                # Store the current frame into frameData
                frameData[currentIndex] = detObj
                currentIndex += 1

            # time.sleep(0.01) # Sampling frequency of 30 Hz
            
        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            # win.close()
            break

def run2():
    pg.exec()

if __name__ == '__main__':
    #Hacemos un thread para la funcion run y que esté en paralelo con el bucle infinito de la GUI
    t1=Thread(target=run)
    t1.start()
    t2 = Thread(target=run2)
    t2.start()
    window.mainloop()
        
