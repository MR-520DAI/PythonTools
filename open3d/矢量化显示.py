import keyboard
import numpy as np
import open3d as o3d

# 创建一个空的点云对象
point_cloud = o3d.geometry.PointCloud()
# 创建一个显示窗口，并将点云添加到窗口中
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud)

# def updateVis(points):
#     global point_cloud, vis
#     point_cloud += points
#     # 更新渲染器
#     vis.update_geometry(point_cloud)
#     vis.poll_events()
#     vis.update_renderer()

if __name__ == "__main__":
    root_dir = "E:\\data\\rgbd\\point\\"
    for i in range(100):
        pcd = o3d.io.read_point_cloud(root_dir + str(i) + "_transform.ply")
        point_cloud += pcd
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
