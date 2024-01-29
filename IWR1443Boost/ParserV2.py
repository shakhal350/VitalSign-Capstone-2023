import math
import struct
from pathlib import Path


#
# Specifically for mmWave SDK version 2.1
#

# Run for debug
# /usr/local/opt/python/bin/python3.7 -m debugpy --wait-for-client --listen 0.0.0.0:5678 parseTLV.py

# CLI Profile Configuration Variables
# startFreq           = 77 #GHz
# chirpMargin_default = 0  #Might change with frame... gonna ignore for now
# numChirpsPerFrame   = 64
# frameDuration       = 100000 #usec or 100 msec
# chirpingTime        = frameDuration - chirpMargin_default
# speedOfLight        = 299792458.0 #m/s

def tlvHeaderDecode(data):
    tlvType, tlvLength = struct.unpack('<2I', data)
    return tlvType, tlvLength


def parseDetectedObjects(data, tlvLength):
    inital_offset = 4
    numDetectedObj, xyzQFormat = struct.unpack('<2H', data[:inital_offset])
    print("\tDetect Obj:\t%d " % (numDetectedObj))
    for i in range(numDetectedObj):
        print("\tObjId:\t%d " % (i))
        rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack('<HhHhhh', data[inital_offset + 12 * i: inital_offset + 12 * i + 12])
        print("\t\tDopplerIdx:\t%d " % (dopplerIdx))
        print("\t\tRangeIdx:\t%d " % (rangeIdx))
        print("\t\tPeakVal:\t%d " % (peakVal))
        print("\t\tX:\t\t%07.3f " % (x * 1.0 / (1 << xyzQFormat)))
        print("\t\tY:\t\t%07.3f " % (y * 1.0 / (1 << xyzQFormat)))
        print("\t\tZ:\t\t%07.3f " % (z * 1.0 / (1 << xyzQFormat)))
        print("\t\tRange:\t\t%07.3fm" % (math.sqrt(pow((x * 1.0 / (1 << xyzQFormat)), 2) + pow((y * 1.0 / (1 << xyzQFormat)), 2) + pow((z * 1.0 / (1 << xyzQFormat)), 2))))
        # print("\t\tVelocity:\t\t%0.3fm/s"%(dopplerIdx * (speedOfLight/ (2 * (startFreq * 1e9) * frameDuration * 1e-6)) )  )


def parseRangeProfile(data, tlvLength):
    looper = int(tlvLength / 2)
    for i in range(looper):
        rangeProfile = struct.unpack('<H', data[2 * i:2 * i + 2])
        print("\tRangeProf[%0.3fm]:\t%07.3fdB " % (i * 0.1249921875, 20 * math.log10(2 ** (rangeProfile[0] / (2 ** 9)))))  # 0.1249921875 is based on profile confg
    print("\tTLVType:\t%d " % (2))


def parseStats(data, tlvLength):
    interProcess, transmitOut, frameMargin, chirpMargin, activeCPULoad, interCPULoad = struct.unpack('<6I', data[:tlvLength])
    print("\tOutputMsgStats:\t%d " % (6))
    print("\t\tChirpMargin:\t%d " % (chirpMargin))
    print("\t\tFrameMargin:\t%d " % (frameMargin))
    print("\t\tInterCPULoad:\t%d " % (interCPULoad))
    print("\t\tActiveCPULoad:\t%d " % (activeCPULoad))
    print("\t\tTransmitOut:\t%d " % (transmitOut))
    print("\t\tInterprocess:\t%d " % (interProcess))


def tlvHeader(data, skip_range=False, skip_stats=False):
    pendingBytes = 29
    while pendingBytes > 28:
        # find start magic
        magic = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        offset = data.find(magic)
        data = data[offset:]
        headerLength = 28
        # Shift data off of Magic
        data = data[8:]
        try:
            version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack('<7I', data[:headerLength])
        except struct.error as e:
            print("Improper TLV structure found: ", (data,))
            print("Error ", e)
            print(pendingBytes)
            break
        print("Packet ID:\t%d " % (frameNum))
        print("Version:\t%x " % (version))
        print("TLV:\t\t%d " % (numTLVs))
        print("Detect Obj:\t%d " % (numObj))
        print("Platform:\t", hex(platform))
        # print("Subframe:\t%d "%(subFrameNum))
        pendingBytes = length - headerLength
        data = data[headerLength:]
        for i in range(numTLVs):
            tlvType, tlvLength = tlvHeaderDecode(data[:8])
            data = data[8:]
            if (tlvType == 1):
                parseDetectedObjects(data, tlvLength)
            elif (tlvType == 2):
                if not skip_range:
                    parseRangeProfile(data, tlvLength)
            elif (tlvType == 6):
                if not skip_stats:
                    parseStats(data, tlvLength)
                # tlvLength = tlvLength + 4
            else:
                print("Unidentified tlv type %d" % (tlvType))
            data = data[tlvLength:]
            pendingBytes -= (8 + tlvLength)
        # data = data[pendingBytes:]


if __name__ == "__main__":
    # fileName = Path('xwr14xx_processed_stream_2020_06_30T16_54_36_103.dat') #Lots of stuff. Not working
    # fileName = Path('xwr14xx_processed_stream_2020_06_26T21_03_42_880.dat') #Few things, working
    # fileName = Path('xwr14xx_processed_stream_2020_07_13T14_11_18_703.dat')
    # fileName = Path('Test.dat')
    fileName = Path(r'C:\Users\Shaya\PycharmProjects\VitalSign-Capstone-2023\IWR1443Boost\data_recorded_from_mmwaveDemoVisualizer.dat')
    rawDataFile = open(fileName, "rb")
    rawData = rawDataFile.read()
    rawDataFile.close()

    tlvHeader(rawData, skip_stats=True, skip_range=False)
    print("End...")
