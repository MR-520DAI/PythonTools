import open3d as o3d
import numpy as np

color_raw = o3d.io.read_image("data\\colmap\\images\\0.jpg")
depth_raw = o3d.io.read_image("data\\colmap\\depth\\0.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw, False)

# 设置相机参数
width = 640
height = 480
fx = 520.0  # 焦距x（单位像素）
fy = 520.0  # 焦距y（单位像素）
cx = 320.0  # 光心x
cy = 240.0  # 光心y

# 创建 PinholeCameraIntrinsic 对象来定义相机参数
cam = o3d.camera.PinholeCameraParameters()
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
cam.intrinsic = intrinsic
cam.extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

all_pcd = o3d.geometry.PointCloud()
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam.intrinsic, cam.extrinsic)
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
all_pcd += pcd
o3d.visualization.draw_geometries([all_pcd])
