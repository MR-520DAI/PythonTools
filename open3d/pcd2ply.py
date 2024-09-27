import open3d as o3d

# pcd = o3d.io.read_point_cloud("E:\\data\\livox\\map\\office-360.pcd")
# o3d.io.write_point_cloud("E:\\data\\livox\\map\\office-360.ply", pcd)

root_dir = "E:\\data\\rgbd\\vector_model\\"

for i in range(10):
    print(i)
    pcd_org = o3d.io.read_point_cloud(root_dir+"point\\"+str(i)+"_org.ply")
    pcd_transform = o3d.io.read_point_cloud(root_dir+"point\\"+str(i)+"_transform.ply")

    o3d.io.write_point_cloud(root_dir+"point1\\"+str(i)+"_org.pcd", pcd_org, write_ascii=True)
    o3d.io.write_point_cloud(root_dir+"point1\\"+str(i)+"_transform.pcd", pcd_transform, write_ascii=True)