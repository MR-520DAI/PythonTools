# -*- coding: utf-8 -*-
import numpy as np

camera_ground_height = 1.65     # 相机与地面的高度（米）
Fx = 721.5377                   # 相机内参
Fy = 721.5377
Cx = 596.5593
Cy = 149.854

# 车轮处的像素坐标，目标检测算法能够完成
image_point = np.matrix([[593], [195], [1]], dtype="float")

camera_k_matrix = np.matrix([[Fx, 0, Cx], [0, Fy, Cy], [0, 0, 1]], dtype="float")
camera_k_matrix_intrinsic = camera_k_matrix.I

camera_point = camera_k_matrix_intrinsic * image_point
print("不包含尺度信息的坐标:")
print(camera_point)

Zc = camera_ground_height / camera_point[1][0]
print("Xc:", camera_point[0][0] * Zc)
print("Yc:", camera_point[1][0] * Zc)
print("Zc:", camera_point[2][0] * Zc)