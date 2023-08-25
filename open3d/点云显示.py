import open3d as o3d
import numpy as np

# 设置相机参数
width = 640
height = 480
fx = 518.9676  # 焦距x（单位像素）
fy = 518.8752  # 焦距y（单位像素）
cx = 320.5550  # 光心x
cy = 237.8842  # 光心y

# 创建 PinholeCameraIntrinsic 对象来定义相机参数
cam = o3d.camera.PinholeCameraParameters()
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
# intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
cam.intrinsic = intrinsic
cam.extrinsic = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

source_color = o3d.io.read_image("data\\ICP\\office\\1.jpg")
source_depth = o3d.io.read_image("data\\ICP\\office\\1.png")
target_color = o3d.io.read_image("data\\ICP\\office\\2.jpg")
target_depth = o3d.io.read_image("data\\ICP\\office\\2.png")

source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth, depth_scale=1250)
target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth, depth_scale=1250)

source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd, cam.intrinsic, cam.extrinsic)
target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd, cam.intrinsic, cam.extrinsic)

# pose0 = np.array([[0.90831,-0.0221838,0.41771,-0.308821],
#                   [0.0467274,0.997724,-0.0486215,0.101917],
#                   [-0.41568,0.0636819,0.907279,0.0762022],
#                   [0.,0.,0.,1.]])
pose0 = np.array([[ 0.95112042, -0.01854228,  0.30826308, -0.01471358],
 [ 0.02485266,  0.99955401, -0.01655684,  0.0017963 ],
 [-0.30781859,  0.02340871,  0.95115706, -0.0121713 ],
 [ 0.,          0.,          0.,          1.        ]])

P = np.array([[-0.6691252516199748],
              [0.3111449528991072],
              [2.9698759148692484],
              [1.]])
print(np.matmul(pose0, P))

pcd_all = o3d.geometry.PointCloud()
pcd_all+=source_pcd
pcd_all+=target_pcd.transform(pose0)

pcd_all.voxel_down_sample(voxel_size=0.1)
o3d.visualization.draw_geometries([pcd_all])