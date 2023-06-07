import open3d as o3d

# 读取点云数据
cloud = o3d.io.read_point_cloud("data/result.pcd")
if cloud is None:
    print("Failed to read file input.pcd")
    exit(1)

# 将网格数据保存到 STL 文件中
o3d.io.write_point_cloud("output.ply", cloud)

print("Done!")
