import os

txtroot = "/home/dzy/c_project/ORB_SLAM3-master/Examples/Stereo/EuRoC_TimeStamps/"
imgroot = "/home/dzy/data/MYdata/20221130/mav0/cam1/data/"

new_name = []
old_name = []

with open(txtroot + "V101.txt", "r") as f:
    data = f.readlines()
    for line in data:
        line = line.strip("\n")
        new_name.append(line)

with open(txtroot + "MYdata20221130.txt", "r") as f:
    data = f.readlines()
    for line in data:
        line = line.strip("\n")
        old_name.append(line)

for i in range(len(old_name)):
    os.rename(imgroot + old_name[i] + ".png", imgroot + new_name[i] + ".png")