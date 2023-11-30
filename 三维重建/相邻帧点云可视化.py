import cv2
import numpy as np
import open3d as o3d

def get_polygon_pixels(vertices, scale=1):
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
    root = "E:\\data\MYData\\CamLidardata\\"
    # 相机参数
    fx = 406.1814  # 焦距x（单位像素）
    fy = 405.7139  # 焦距y（单位像素）
    cx = 311.5565  # 光心x
    cy = 250.0853  # 光心y

    Planes = [
        [[0.20879321132478587, 0.11817739422434466, -0.9707932315375026, 1.6985628794316876]],
        [[-0.0223137,-0.0543021,0.998275,-1.79477]]
    ]
    [0.5437833152069513,0.04477061508915761,-0.838030606915449,1.7050009520395548]
    Vertices = [
        [[[2, 2], [620, 2], [620, 369], [2, 369]]],
        [[[2, 2], [620, 2], [620, 369], [2, 369]]],
    ]
    print(Vertices[0])
    Ts = [
        [np.array([[1.,0.,0.,0.],
                [0.,1.,0.,0.],
                [0.,0.,1.,0.],
                [0.,0.,0.,1.]])],
        [np.array([[ 0.975656 , -0.104127 ,  -0.19301,-0.131942],
 [ 0.0937399 ,  0.993651, -0.0622161 ,0.0439829],
 [ 0.198262 , 0.0426088 ,  0.979222, -0.122126],
 [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])],
    ]

    [[0.979745,-0.00525998,-0.20018,-0.0507371,],
    [0.00120885,0.999792,-0.0203543,0.0106038,],
    [0.200245,0.0197001,0.979548,-0.00214268,],
    [0.,0.,0.,1.]]

    T = np.array([[1.,0.,0.,0.],
                   [0.,1.,0.,0.],
                   [0.,0.,1.,0.],
                   [0.,0.,0.,1.]])
    
    pcd_all = o3d.geometry.PointCloud()
    for i in range(0, 2, 1):
        T = np.matmul(T, Ts[i][0])
        img = cv2.imread(root + str(i+5) + ".jpg")
        for vertice, plane in zip(Vertices[i], Planes[i]):
            pixels = get_polygon_pixels(vertice)
            pcd = GetPcd(img, pixels, plane, fx, fy, cx, cy, T)
            pcd_all += pcd
    
    pcd_all.voxel_down_sample(voxel_size=0.1)
    o3d.visualization.draw_geometries([pcd_all])
    o3d.io.write_point_cloud("output.ply", pcd_all)