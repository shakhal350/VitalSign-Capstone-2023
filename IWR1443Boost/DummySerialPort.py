class DummySerialPort:
    def __init__(self, port, baudrate, bytesize, parity, stopbits, timeout, xonxoff, rtscts, dsrdtr):
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.xonxoff = xonxoff
        self.rtscts = rtscts
        self.dsrdtr = dsrdtr
        print(f"Initialized dummy serial port: {port} with baudrate: {baudrate}")

    def open(self):
        print(f"Opening dummy serial port: {self.port}")

    def close(self):
        print(f"Closing dummy serial port: {self.port}")

    def read(self, size=1):
        print(f"Reading {size} bytes from dummy serial port: {self.port}")
        return b'\x00' * size  # Return dummy data

    def write(self, data):
        print(f"Writing to dummy serial port: {self.port}")
        return len(data)  # Return the length of data as if it was written

    def isOpen(self):
        return True