import cv2
import torch
import math
import numpy as np
import kornia as K
import open3d as o3d
import kornia.feature as KF
from scipy.spatial.transform import Rotation

def Get3DPointsOnPlane(pixel, plane, fx, fy, cx, cy):
    """根据平面方程恢复像素坐标为3D坐标

    Args:
        pixel (list): 像素坐标
        plane (float): 平面方程参数
        fx (float): 相机内参
        fy (float): 相机内参
        cx (float): 相机内参
        cy (float): 相机内参

    Returns:
        list: 3D坐标
    """
    x = pixel[0]
    y = pixel[1]
    X = (x-cx) / fx
    Y = (y-cy) / fy
    scale = (-plane[3]) / (plane[0]*X + plane[1]*Y + plane[2])
    X = X * scale
    Y = Y * scale
    Z = scale
    return [X, Y, Z]

def CamToPixel(point, K):
    p = np.matmul(K, point)
    p /= p[2][0]
    return p

def CulAngle(v1, v2):
    dot_product = np.dot(v1, v2)  
    norm1 = np.linalg.norm(v1)  
    norm2 = np.linalg.norm(v2) 
    cos_angle = dot_product / (norm1 * norm2)
    angle = np.arccos(cos_angle)
    return angle

def PixelToCam(point, K):
    K_inv = np.linalg.inv(K)
    p = np.matmul(K_inv, point)
    return p

def GetPcd(img1_path, Kmat):
    img = cv2.imread(img1_path)
    points = []
    colors = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            pixel = np.array([[x], [y], [1.0]], dtype=np.float32)
            point = PixelToCam(pixel, Kmat)
            color = [img[y][x][2]/255, img[y][x][1]/255, img[y][x][0]/255]
            points.append([point[0][0], point[1][0], point[2][0]])
            colors.append(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    return pcd

def estimate_rotation(plane1, plane2):
    n1 = plane1[:3]  # 法向量1
    n2 = plane2[:3]  # 法向量2
    R = np.dot(n2, n1) / np.linalg.norm(n1)**2
    return R

# 估计平移向量
def estimate_translation(plane1, plane2):
    n1 = plane1[:3]  # 法向量1
    n2 = plane2[:3]  # 法向量2
    d1 = plane1[3]  # 距离1
    d2 = plane2[3]  # 距离2
    t = (d2 - d1) * n1 / np.linalg.norm(n1)**2
    return t

def plane_distance(plane1, plane2):
    A1, B1, C1, D1 = plane1
    A, B, C, D = plane2

    x0, y0 = 2.0, 1.0
    z0 = -(A1*x0 + B1*y0+D1) / C1
    distance = abs(A * x0 + B * y0 + C * z0 + D) / math.sqrt(A**2 + B**2 + C**2)

    return distance

if __name__ == "__main__":
    # 相机参数
    fx = 406.1814  # 焦距x（单位像素）
    fy = 406.1814  # 焦距y（单位像素）
    cx = 311.5565  # 光心x
    cy = 250.0853  # 光心y
    Kmat = np.array([[fx, 0., cx],
                  [0., fy, cy],
                  [0., 0., 1.]])
    root = "E:\\data\\MYData\\CamLidardata\\"

    plane1 = [0.178276, -0.106748, 0.978173, -1.52979]
    plane2 = [0.984522, 0.0038439, -0.175218, 1.79639]
    plane3 = [0.180788, -0.057171, 0.981859, 1.90062]

    v1 = np.array([plane1[0], plane1[1], plane1[2]])
    v2 = np.array([plane2[0], plane2[1], plane2[2]])
    v3 = np.array([plane3[0], plane3[1], plane3[2]])

    print(CulAngle(v1, v2) * 180 / 3.14159)
    print(CulAngle(v3, v2) * 180 / 3.14159)