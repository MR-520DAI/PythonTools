import open3d as o3d
import numpy as np
import quaternion

# 设置相机参数
width = 1280
height = 720
fx = 691.9569  # 焦距x（单位像素）
fy = 691.8336  # 焦距y（单位像素）
cx = 641.0734  # 光心x
cy = 357.1790  # 光心y
cam = o3d.camera.PinholeCameraParameters()
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
cam.intrinsic = intrinsic
cam.extrinsic = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

root_dir = "E:\\data\\rgbd\\vector_model\\"

all_pcd = o3d.geometry.PointCloud()

T_c_l = np.array([[0.,-1.,0.,0.],
                  [0.,0.,-1.,-0.06],
                  [1.,0.,0.,-0.063],
                  [0.,0.,0.,1.]])
T_l_c = np.linalg.inv(T_c_l)
T_l_i = np.array([[1.,0.,0.,-0.011],
                  [0.,1.,0.,-0.02329],
                  [0.,0.,1.,0.04412],
                  [0.,0.,0.,1.]])
T_i_l = np.linalg.inv(T_l_i)
T_i_c = np.matmul(T_i_l, T_l_c)

pcd_points = o3d.geometry.PointCloud()
for i in range(0, 1, 1):
    color_raw = o3d.io.read_image(root_dir + "color\\" + str(i) + ".jpg")
    depth_raw = o3d.io.read_image(root_dir + "depth\\" + str(i) + ".png")
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1000, depth_trunc=5)
    print(root_dir + "color\\" + str(i) + ".jpg")

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam.intrinsic, cam.extrinsic)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd_points.points = pcd.points
    o3d.io.write_point_cloud(root_dir + "point\\" + str(i) + "_org.pcd", pcd_points)

    with open(root_dir + "odom\\" + str(i) + ".txt", "r") as f:
        pose_data = f.readline()
        pose_data = pose_data.strip("\n").split(",")
        q = np.quaternion(float(pose_data[0]), float(pose_data[1]), float(pose_data[2]), float(pose_data[3]))
        rotation_matrix = quaternion.as_rotation_matrix(q)
        t = np.array([float(pose_data[4]), float(pose_data[5]), float(pose_data[6])])
        T_w_i = np.eye(4)  # 创建一个单位矩阵作为初始变换矩阵
        T_w_i[:3, :3] = rotation_matrix  # 将旋转矩阵部分填入变换矩阵的旋转部分
        T_w_i[:3, 3] = t  # 将平移向量填入变换矩阵的平移部分
        T_w_c = np.matmul(T_w_i, T_i_c)
    print(i,"--T_w_c:",T_w_c)
    pcd.transform(T_w_c)
    pcd_points.points = pcd.points
    o3d.io.write_point_cloud(root_dir + "point\\" + str(i) + "_transform.pcd", pcd_points)
    
    all_pcd += pcd
    # all_pcd.voxel_down_sample(voxel_size=0.5)

all_pcd = all_pcd.voxel_down_sample(voxel_size=0.02)
o3d.io.write_point_cloud("x.ply", all_pcd)
o3d.visualization.draw_geometries([all_pcd])