import numpy as np
import open3d as o3d

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
target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

# source_pcd = o3d.io.read_point_cloud("data\\ICP\\demo\\cloud_bin_0.pcd")
# target_pcd = o3d.io.read_point_cloud("data\\ICP\\demo\\cloud_bin_1.pcd")
# trans_init_demo = np.asarray([[0.862, 0.011, -0.507, 0.5],
#                          [-0.139, 0.967, -0.215, 0.7],
#                          [0.487, 0.255, 0.835, -1.4],
#                          [0.0, 0.0, 0.0, 1.0]])

trans_init = np.asarray([[1, 0., -0., 0.],
                         [-0, 1., -0, 0.],
                         [0, 0, 1., -0.],
                         [0.0, 0.0, 0.0, 1.0]])

icp_result = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, 0.02, trans_init, 
                                                         o3d.pipelines.registration.TransformationEstimationPointToPlane())

transformation_matrix = icp_result.transformation

# transformation_matrix = np.asarray([[0.934721, -0.0256788, 0.354452, -0.017965],
# [0.0326621, 0.999372, -0.013732, 0.00191247],
# [-0.353877, 0.0244128, 0.934973, -0.0187198],
# [0, 0, 0, 1],])
print(transformation_matrix)

# Transform the source point cloud
target_pcd.transform(transformation_matrix)

# Merge the transformed source point cloud with the target point cloud
merged_pcd = target_pcd + source_pcd

# Visualize the merged point cloud
o3d.visualization.draw_geometries([merged_pcd])