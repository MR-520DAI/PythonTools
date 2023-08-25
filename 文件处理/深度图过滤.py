import cv2
import numpy as np

img = cv2.imread("E:\\data\\rgbd\\office\\depth\\1685686754.843132.png", cv2.IMREAD_UNCHANGED)

img_np = np.array(img)
count = np.count_nonzero(img_np == 0)
print(count)

root_rgb = "E:\\data\\rgbd\\office\\rgb\\"
root_depth = "E:\\data\\rgbd\\office\\depth\\"

num = 0
with open("E:\\data\\rgbd\\office\\associate_win.txt", "r") as f:
    data = f.readlines()
    for line in data:
        line = line.strip("\n").split(" ")
        img_depth = cv2.imread(root_depth + line[3], cv2.IMREAD_UNCHANGED)
        img_depth_np = np.array(img_depth)
        zero_count = np.count_nonzero(img_depth_np == 0)
        #print(line[1], ":", zero_count, " 比例:", zero_count / (640*480))
        if (zero_count / (640*480)) > 0.8:  # 若无效值太多则过滤
            print(line[1], "    ", line[3])
        else:
            continue


