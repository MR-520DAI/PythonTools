import open3d as o3d
import numpy as np
import os

def read_xyz_file(file_path):
    with open(file_path, 'r') as file:
        points = []
        data = file.readlines()
        for line in data:
            xyz = line.strip("\n").split(",")
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
            points.append([x,y,z])


    return np.array(points, dtype=np.float32)

if __name__ == "__main__":
    root = "E:\\py_project\\PythonTools\\open3d\\data\\xyz\\V2\\"
    file_list = os.listdir(root)
    for name in file_list:
        points_path = root + name

        points = read_xyz_file(points_path)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        o3d.io.write_point_cloud(root + name.split(".")[0]+".ply", point_cloud)
        print(name)
        # break