import struct
from pathlib import Path


#
# TODO 1: (NOW FIXED) Find the first occurrence of magic and start from there
# TODO 2: Warn if we cannot parse a specific section and try to recover
# TODO 3: Remove error at end of file if we have only fragment of TLV
#

def tlvHeaderDecode(data):
    tlvType, tlvLength = struct.unpack('2I', data)
    return tlvType, tlvLength


def parseDetectedObjects(data, tlvLength):
    numDetectedObj, xyzQFormat = struct.unpack('2H', data[:4])
    print("\tDetect Obj:\t%d " % (numDetectedObj))
    for i in range(numDetectedObj):
        print("\tObjId:\t%d " % (i))
        rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack('3H3h', data[4 + 12 * i:4 + 12 * i + 12])
        print("\t\tDopplerIdx:\t%d " % (dopplerIdx))
        print("\t\tRangeIdx:\t%d " % (rangeIdx))
        print("\t\tPeakVal:\t%d " % (peakVal))
        print("\t\tX:\t\t%07.3f " % (x * 1.0 / (1 << xyzQFormat)))
        print("\t\tY:\t\t%07.3f " % (y * 1.0 / (1 << xyzQFormat)))
        print("\t\tZ:\t\t%07.3f " % (z * 1.0 / (1 << xyzQFormat)))


def parseRangeProfile(data, tlvLength):
    for i in range(256):
        rangeProfile = struct.unpack('H', data[2 * i:2 * i + 2])
        print("\tRangeProf[%d]:\t%07.3f " % (i, rangeProfile[0] * 1.0 * 6 / 8 / (1 << 8)))
    print("\tTLVType:\t%d " % (2))


def parseStats(data, tlvLength):
    interProcess, transmitOut, frameMargin, chirpMargin, activeCPULoad, interCPULoad = struct.unpack('6I', data[:24])
    print("\tOutputMsgStats:\t%d " % (6))
    print("\t\tChirpMargin:\t%d " % (chirpMargin))
    print("\t\tFrameMargin:\t%d " % (frameMargin))
    print("\t\tInterCPULoad:\t%d " % (interCPULoad))
    print("\t\tActiveCPULoad:\t%d " % (activeCPULoad))
    print("\t\tTransmitOut:\t%d " % (transmitOut))
    print("\t\tInterprocess:\t%d " % (interProcess))


def tlvHeader(data):
    print("Start...tlvHeader")
    while data:
        headerLength = 36
        try:
            magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack('Q7I', data[:headerLength])
        except:
            print("Improper TLV structure found: ", (data,))
            break
        print("Packet ID:\t%d " % (frameNum))
        print("Version:\t%x " % (version))
        print("TLV:\t\t%d " % (numTLVs))
        print("Detect Obj:\t%d " % (numObj))
        print("Platform:\t%X " % (platform))
    if version > 0x01000005:
        subFrameNum = struct.unpack('I', data[36:40])[0]
        headerLength = 40
        print("Subframe:\t%d " % (subFrameNum))
        pendingBytes = length - headerLength
        data = data[headerLength:]
        for i in range(numTLVs):
            tlvType, tlvLength = tlvHeaderDecode(data[:8])
            data = data[8:]
            if (tlvType == 1):
                parseDetectedObjects(data, tlvLength)
            elif (tlvType == 2):
                parseRangeProfile(data, tlvLength)
            elif (tlvType == 6):
                parseStats(data, tlvLength)
            else:
                print("Unidentified tlv type %d" % (tlvType))
            data = data[tlvLength:]
            pendingBytes -= (8 + tlvLength)
        data = data[pendingBytes:]
        yield length, frameNum


if __name__ == "__main__":

    fileName = Path(r'C:\Users\Shaya\PycharmProjects\VitalSign-Capstone-2023\IWR1443Boost\data_recorded_from_mmwaveDemoVisualizer.dat')
    rawDataFile = open(fileName, "rb")
    rawData = rawDataFile.read()
    rawDataFile.close()
    magic = b'\x02\x01\x04\x03\x06\x05\x08\x07'
    offset = rawData.find(magic)
    rawData = rawData[offset:]
    for length, frameNum in tlvHeader(rawData):
        print("End...")
