import cv2
import numpy as np
import open3d as o3d

def get_polygon_pixels(vertices, scale=2):
    # 获取四边形顶点的坐标
    p1, p2, p3, p4 = vertices
    
    # 确定四边形的最小和最大边界
    x_min = int(min(p1[0], p2[0], p3[0], p4[0]))
    x_max = int(max(p1[0], p2[0], p3[0], p4[0]))
    y_min = int(min(p1[1], p2[1], p3[1], p4[1]))
    y_max = int(max(p1[1], p2[1], p3[1], p4[1]))

    # 创建用于存储所有像素坐标的列表
    pixel_coords = []

    # 遍历四边形内的所有像素点
    for y in range(y_min, y_max + 1, scale):
        for x in range(x_min, x_max + 1, scale):
            if is_point_inside_polygon(x, y, vertices):
                pixel_coords.append((x, y))

    return pixel_coords

def is_point_inside_polygon(x, y, vertices):
    # 判断点是否在四边形内部
    p1, p2, p3, p4 = vertices
    
    # 根据点是否在四条边的同一侧来判断点是否在四边形内部
    return (
        is_on_same_side(x, y, p1, p2, p3) 
        and is_on_same_side(x, y, p2, p3, p4)
        and is_on_same_side(x, y, p3, p4, p1)
        and is_on_same_side(x, y, p4, p1, p2)
    )

def is_on_same_side(x, y, p1, p2, p):
    # 检查点p是否在p1p2直线的同一侧
    return (y - p1[1]) * (p2[0] - p1[0]) - (x - p1[0]) * (p2[1] - p1[1]) >= 0

def Get3DPointsOnPlane(pixels, Plane, img, fx, fy, cx, cy):
    Points = []
    Colors = []
    for i in pixels:
        x = i[0]
        y = i[1]
        X = (x-cx) / fx
        Y = (y-cy) / fy
        scale = (-Plane[3]) / (Plane[0]*X + Plane[1]*Y + Plane[2])
        X = X * scale
        Y = Y * scale
        Z = scale
        Points.append([X, Y, Z])
        Colors.append([img[y][x][2]/255, img[y][x][1]/255, img[y][x][0]/255])
    return Points, Colors

def GetPcd(img, pixels, plane, fx, fy, cx, cy, T):
    points, colors = Get3DPointsOnPlane(pixels, plane, img, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(T)

    return pcd

if __name__ == "__main__":
    # 相机参数
    fx = 518.9676  # 焦距x（单位像素）
    fy = 518.8752  # 焦距y（单位像素）
    cx = 320.5550  # 光心x
    cy = 237.8842  # 光心y

    p1 = (290, 2)
    p2 = (630, 2)
    p3 = (630, 476)
    p4 = (300, 476)
    vertices12 = [(178, 3), (636, 3), (636, 476), (189, 476)]
    img12 = cv2.imread("data\\img\\2.jpg")
    plane12 = [0.375814, -0.0428255, 0.925705, -1.50254]
    pixels12 = get_polygon_pixels(vertices12)
    p1 = (2, 2)
    p2 = (289, 2)
    p3 = (299, 476)
    p4 = (2, 476)
    vertices12_1 = [(3, 3), (179, 3), (189, 476), (3, 476)]
    plane12_1 = [0.927755, 0.00108734, -0.373187, 1.12946]
    pixels12_1 = get_polygon_pixels(vertices12_1)
    T12 = np.array([[1.,0.,0.,0.],
                   [0.,1.,0.,0.],
                   [0.,0.,1.,0.],
                   [0.,0.,0.,1.]])
    pcd12 = GetPcd(img12, pixels12, plane12, fx, fy, cx, cy, T12)
    pcd12_1 = GetPcd(img12, pixels12_1, plane12_1, fx, fy, cx, cy, T12)

    pcd_all = o3d.geometry.PointCloud()
    pcd_all += pcd12
    pcd_all += pcd12_1
    pcd_all.voxel_down_sample(voxel_size=0.1)
    o3d.visualization.draw_geometries([pcd_all])