import os

filename = os.listdir("/workspace/Ultra-Fast-Lane-Detection/data/CULane/driver_37_30frame")

for i in filename:
    print("/workspace/LaneDect/txt_result/driver_37_30frame/" + i)
    os.mkdir("/workspace/LaneDect/txt_result/driver_37_30frame/" + i)