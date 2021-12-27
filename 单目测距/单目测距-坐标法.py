# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

camera_ground_height = 1526
Fx = 1103.709
Fy = 997.642
Cx = 956.093
Cy = 562.057
Vx = 856
Vy = 559

image_point = np.matrix([[935], [668], [1]], dtype="float")

camera_k_matrix = np.matrix([[Fx, 0, Cx], [0, Fy, Cy], [0, 0, 1]], dtype="float")
camera_k_matrix_intrinsic = camera_k_matrix.I

# print(camera_k_matrix_intrinsic)

org_camera_point = camera_k_matrix_intrinsic * image_point
print(org_camera_point)

theta = np.arctan((Cy-Vy)/Fy)
w = np.arctan(((Cx - Vx)/Fx) * np.cos(np.arctan((Cy-Vy)/Fy)))

pitch_matrix = np.matrix([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
yaw_matrix = np.matrix([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0, 0, 1]])
# print(pitch_matrix)

rotate_point = pitch_matrix.I * org_camera_point
print(pitch_matrix.I)
print('----------------')
# rotate_point = yaw_matrix.I * rotate_point
# print(yaw_matrix.I)
scale = camera_ground_height / rotate_point[1][0]

rotate_point[0][0] = rotate_point[0][0] * scale
rotate_point[1][0] = rotate_point[1][0] * scale
rotate_point[2][0] = rotate_point[2][0] * scale

print(rotate_point)

# x1 = (2229-1307.5)/1853.5
# y1 = (1533-1023.7)/1853.5
# r2 = x1 * x1 + y1 * y1
# k1 = -0.13
# k2 = -0.45
# p1 = -0.00085
# p2 = 0.002
# k3 = 0.44
# x2 = x1 * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * x1 * y1 + p2 * (r2 + 2*x1*x1)
# y2 = y1 * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + p1 * (r2 + 2*y1*y1) + 2*p2*x1*y1
#
# x3 = x2 * 1853.5 + 1307.5
# y3 = y2 * 1853.5 + 1023.7
#
# print(x3)
# print(y3)
