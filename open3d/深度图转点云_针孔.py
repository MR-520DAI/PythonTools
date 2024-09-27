import open3d as o3d
import numpy as np

W = 256
H = 192
fx = 120.28993958745004
cx = 132.0095238469846
fy = 120.43877051179346
cy = 96.2253386563752
k1 = -1.4662624605194327e-01
k2 = -1.9759328248453203e-01
p1 = -1.4127488172676100e-02
p2 = 1.5695179151999530e-03
k3 = 1.9972434899732325e-01

depth = []
depth_path = "E:\\data\\lingguang\\Record_FHR_2024_05_14_09_39_06\\Depth_frameid_0_2914169923.txt"
with open(depth_path, "r") as f:
  datas = f.readlines()
  for data in datas:
    data = data.strip("\n").split(",")
    depth.append(data[0:256])

points = []
for i in range(H):
  for j in range(W):
    d = float(depth[191-i][255-j])
    if d <= 0.5 or d > 10.0:
      continue
    x_norm = (j-cx) / fx
    y_norm = (i-cy) / fy
    r = x_norm*x_norm + y_norm*y_norm
    x_distort = x_norm*(1+k1*r+k2*r*r+k3*r*r*r) + 2*p1*x_norm*y_norm + p2*(r+2*x_norm*x_norm)
    y_distort = y_norm*(1+k1*r+k2*r*r+k3*r*r*r) + p1*(r+2*y_norm*y_norm) + 2*p2*x_norm*y_norm
    X = x_distort*d
    Y = y_distort*d
    Z = d
    points.append([X,Y,Z])

points = np.array(points, dtype=np.float32)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("E:\\data\\lingguang\\Record_FHR_2024_05_14_09_39_06\\Depth_frameid_0_2914169923.ply", point_cloud)