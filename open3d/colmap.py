import os
import open3d as o3d
import numpy as np

pose = []
with open("data\\colmap\\data.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        data = line.strip().split(",")
        p = np.array([[float(data[0]), float(data[1]), float(data[2]), float(data[3])],
                      [float(data[4]), float(data[5]), float(data[6]), float(data[7])],
                      [float(data[8]), float(data[9]), float(data[10]), float(data[11])],
                      [0.,0.,0.,1.]])

        print(p)
        pose.append(np.linalg.inv(p))

root = "data\\colmap\\"
colorName = os.listdir(root+"images")
depthName = os.listdir(root+"depth")

width = 640
height = 480
fx = 520.0  # 焦距x（单位像素）
fy = 520.0  # 焦距y（单位像素）
cx = 320.0  # 光心x
cy = 240.0  # 光心y
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

all_pcd = o3d.geometry.PointCloud()

for i in range(len(pose)):
    if i == 2:
        break
    color_raw = o3d.io.read_image(root+"images\\"+colorName[i])
    depth_raw = o3d.io.read_image(root+"depth\\"+depthName[i])
    rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw, False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    all_pcd += pcd.transform(pose[i])

o3d.visualization.draw_geometries([all_pcd])