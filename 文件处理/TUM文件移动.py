import os
import shutil

num = 0
root = "/home/dzy_lap/data/TUM/rgbd_dataset_freiburg3_long_office_household/"

with open(root + "associate.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n").split(" ")
        old_color_path = os.path.join(root, line[1])
        new_color_path = os.path.join(root, "colorimg", str(num).zfill(6)+".png")
        shutil.copy(old_color_path, new_color_path)

        old_depth_path = os.path.join(root, line[3])
        new_depth_path = os.path.join(root, "depthimg", str(num).zfill(6)+".png")
        shutil.copy(old_depth_path, new_depth_path)
        num += 1
        print(num)
        if num == 599:
            break


