import os
import cv2
import rosbag
import pcl_ros
from sensor_msgs.msg import PointCloud2
import numpy as np


# imgPath = "/home/dzy_lap/c_project/test/build/1.png"

# img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

# print(img[471][20] / 5000.0)

def GeneratePointCloud(img, fx, fy, cx, cy):
    rows, cols = img.shape[:2]

    u, v = np.meshgrid(range(cols), range(rows))
    z = img / 5000.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).transpose()
    # cloud = pcl.PointCloud()

fx = 535.4
fy = 539.2
cx = 320.1
cy = 247.6

root = "/home/dzy_lap/data/TUM/rgbd_dataset_freiburg3_long_office_household/depth/"
imgName = os.listdir(root)
# print(imgName)
for i in imgName:
    img = cv2.imread(root + i, cv2.IMREAD_UNCHANGED)

