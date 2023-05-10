import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("data\\integrated.ply")
pcd1 = o3d.io.read_point_cloud("data\\integrated.ply")

pcd += pcd1
# p1 = np.asarray(pcd.points)
# print(p1[0])
# pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0.1, 1]])
# p2 = np.asarray(pcd.points)
# print(p2[0])

pcd.voxel_down_sample(voxel_size=0.1)
o3d.visualization.draw_geometries([pcd])