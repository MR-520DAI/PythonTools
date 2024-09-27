import open3d as o3d
import numpy as np

root = "E:\\py_project\\PythonTools\\open3d\\data\\vector\\"
pcd = o3d.io.read_point_cloud(root+"office_seg.ply")

with open(root+"index.txt") as f:
    lines = f.readlines()
    seq = 0
    for line in lines:
        points = []
        plane_points = o3d.geometry.PointCloud()

        line = line.strip("\n").split(",")
        line = line[0:-2]
        for id in line:
            points.append([pcd.points[int(id)][0], pcd.points[int(id)][1], pcd.points[int(id)][2]])
        points = np.array(points)
        plane_points.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(root+str(seq)+"_plane.pcd", plane_points)
        seq += 1
        print(seq)