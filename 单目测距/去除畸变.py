# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

img = cv2.imread("2.png")
# 左相机参数
camera_matrix = np.matrix([[1.1636958274466082e+02, 0., 1.2503408203340558e+02], [0, 1.1636958274466082e+02, 8.0737219011365440e+01], [0, 0, 1]])
distortion_coefficients = np.matrix([[-1.4662624605194327e-01], [-1.9759328248453203e-01], [-1.4127488172676100e-02], [1.5695179151999530e-03], [1.9972434899732325e-01]])
# 右相机参数
# camera_matrix = np.matrix([[2.2003406982612023e+03, 0., 9.9017948552151142e+02], [0, 2.1989773275557795e+03, 5.6700409473508148e+02], [0., 0., 1.]])
# distortion_coefficients = np.matrix([[-8.2266852133274293e-01], [1.2337344941131776e+00], [-3.4474476305577989e-03], [-5.2138110691419505e-03], [-1.1906641291862861e+00]])

dst = cv2.undistort(img, camera_matrix, distortion_coefficients)

cv2.imwrite("undis.png", dst)


# fx=4.0618143898534169e+02
# fy=4.0571395526183096e+02
# cx=3.1155656245286508e+02
# cy=2.5008535916730622e+02
# k1=4.3371813966608198e-02
# k2=-5.8758034905895609e-02
# p1=-1.9981874444658016e-03
# p2=-2.9747840590222803e-05
# k3=0.0

# img_undis = np.zeros((640, 480, 1))

# u_sum = []
# v_sum = []
# for v in range(img_undis.shape[0]):
#     for u in range(img_undis.shape[1]):
#         x1 = (u-cx) / fx
#         y1 = (v-cy) / fy
#         r2 = x1 * x1 + y1 * y1
#         x2 = x1 * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * x1 * y1 + p2 * (r2 + 2*x1*x1)
#         y2 = y1 * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + p1 * (r2 + 2*y1*y1) + 2*p2*x1*y1
#         u_dis = x2 * fx + cx
#         v_dis = y2 * fy + cy
#         if u_dis >= 0 and v_dis >= 0 and u_dis <= img_undis.shape[1] and v_dis <= img_undis.shape[0]:
#             img_undis[v][u] = 0 # img[int(v_dis)][int(u_dis)]
#             u_sum.append(abs(u-u_dis))
#             v_sum.append(abs(v-v_dis))
#         else:
#             img_undis[v][u] = 0
# print(sum(u_sum) / len(u_sum))
# print(sum(v_sum) / len(v_sum))
# cv2.imwrite("data/tsfex.jpg", img_undis)
