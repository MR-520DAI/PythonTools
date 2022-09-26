import cv2
import numpy as np

fx = 3.54136417e+03
fy = 3.54136417e+03
cx = 5.79143124e+02
cy = 3.66899548e+02
base = 1.7590876095382200e+02

# 消失点坐标计算俯仰角,偏航角
pitch = np.arctan((cy-444)/fy)
yaw = -np.arctan((cx-671)/fx * np.cos(pitch))
print("pitch:",pitch)
print("yaw:",yaw)

disp = 15
Z = fx * base / disp
X = (485-cx) * Z / fx
Y = (99-cy) * Z / fy
point = np.matrix([[X], [Y], [Z]])

print("校正前：")
print(point)

pitch_matrix = np.matrix([[1, 0, 0], [0, np.cos(pitch), np.sin(pitch)], [0, -np.sin(pitch), np.cos(pitch)]])
yaw_matrix = np.matrix([[np.cos(yaw), 0., -np.sin(yaw)], [0., 1., 0.], [np.sin(yaw), 0., np.cos(yaw)]])

print("校正后：")
print(yaw_matrix * pitch_matrix * point)
