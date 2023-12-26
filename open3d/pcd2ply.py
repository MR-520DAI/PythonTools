import open3d as o3d

pcd = o3d.io.read_point_cloud("E:\\data\\livox\\map\\office-360.pcd")
o3d.io.write_point_cloud("E:\\data\\livox\\map\\office-360.ply", pcd)
