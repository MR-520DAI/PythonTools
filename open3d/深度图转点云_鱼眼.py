import open3d as o3d
import numpy as np
import cv2

W = 256
H = 192
fx = 121.35078337502344
cx = 134.76764230651165
fy = 121.23663274751947
cy = 97.95035796932528
k1 = 0.09515957040074036
k2 = -0.3171749462967648
k3 = 0.409069961617535
k4 = -0.17746697055901417

depth = []
depth_path = "E:\\data\\lingguang\\calib\\Depth_frameid_0_497397404.txt"
with open(depth_path, "r") as f:
  datas = f.readlines()
  for data in datas:
    data = data.strip("\n").split(",")
    depth.append(data[0:256])

DIM=(256, 192)
K=np.array([[121.35078337502344, 0.0, 134.76764230651165], [0.0, 121.23663274751947, 97.95035796932528], [0.0, 0.0, 1.0]])
D=np.array([[0.09515957040074036], [-0.3171749462967648], [0.409069961617535], [-0.17746697055901417]])

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

points_dis = []
for i in range(H):
  for j in range(W):
    x_index = map1[i][j][0]
    y_index = map1[i][j][1]
    if x_index < 0. or x_index >= (W-1) or y_index < 0. or y_index >= (H-1):
      continue
    d = float(depth[191-y_index][255-x_index])
    if d <= 0.5 or d > 15.0:
      continue
    x = (j-cx) / fx
    y = (i-cy) / fy
    X = x*d
    Y = y*d
    Z = d
    points_dis.append([X,Y,Z])
    
points_dis = np.array(points_dis, dtype=np.float32)
point_cloud_dis = o3d.geometry.PointCloud()
point_cloud_dis.points = o3d.utility.Vector3dVector(points_dis)
o3d.io.write_point_cloud("E:\\data\\lingguang\\calib\\Depth_frameid_0_497397404_dis.ply", point_cloud_dis)

points = []
for i in range(H):
  for j in range(W):
    
    d = float(depth[191-i][255-j])
    if d <= 0.5 or d > 15.0:
      continue
    x = (j-cx) / fx
    y = (i-cy) / fy
    X = x*d
    Y = y*d
    Z = d
    points.append([X,Y,Z])
    
points = np.array(points, dtype=np.float32)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("E:\\data\\lingguang\\calib\\Depth_frameid_0_497397404.ply", point_cloud)