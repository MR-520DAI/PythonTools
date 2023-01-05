import time
import serial

ser = serial.Serial(
    port = "/dev/ttyUSB0",
    baudrate = 115200,
    bytesize = 8,
    parity = "N",
    stopbits = 1,
)

flag = ser.is_open
print(flag)

# 停止雷达
ser.write("stopTracking\r\n".encode('ASCII'))
time.sleep(0.20)
print(ser.read_all())

# 启动雷达
ser.write("startTracking\r\n".encode('ASCII'))
time.sleep(0.20)
print(ser.read_all())

dis = 0
num = 1
try:
    while True:
        data = ser.read()
        if data == b'M':
            data_all = ""
            data_all += data.decode("UTF-8")
        elif data != b'\n':
            data_all += data.decode("UTF-8")
        else:
            # print("距离:")
            # print(data_all.split(" ")[-1])
            dis += float(data_all.split(" ")[-1])
            # print(dis)
            num += 1
        
        if num == 21:
            print(dis / 20)
            num = 1
            dis = 0

except KeyboardInterrupt:
    # 停止雷达
    ser.write("stopTracking\r\n".encode('ASCII'))
    time.sleep(0.20)
    print(ser.read_all())

# 停止雷达
ser.write("stopTracking\r\n".encode('ASCII'))
time.sleep(0.20)
print(ser.read_all())