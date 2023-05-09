import open3d as o3d
import numpy as np

color_raw = o3d.io.read_image("data\\color.png")
depth_raw = o3d.io.read_image("data\\depth.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw)

# 设置相机参数
width = 640
height = 480
fx = 517.306408  # 焦距x（单位像素）
fy = 516.469215  # 焦距y（单位像素）
cx = 318.643040  # 光心x
cy = 255.313989  # 光心y

k1 = 0.26383
k2 = -0.953104
p1 = -0.005358
p2 = 0.002628
k3 = 1.163314

# 创建 PinholeCameraIntrinsic 对象来定义相机参数
cam = o3d.camera.PinholeCameraParameters()
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
cam.intrinsic = intrinsic
cam.extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam.intrinsic, cam.extrinsic)
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])
