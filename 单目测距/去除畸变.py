# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

img = cv2.imread("../data/disp/right2.bmp")
# 左相机参数
camera_matrix = np.matrix([[2.2106200505622032e+03, 0., 9.1430401927216883e+02], [0, 2.2053796021233798e+03, 5.8560607719142990e+02], [0, 0, 1]])
distortion_coefficients = np.matrix([[-1.0027279724022786e+00], [2.1424396386192317e+00], [2.4726513064196013e-03], [1.7457338727093409e-02], [-3.0020294496294349e+00]])
# 右相机参数
# camera_matrix = np.matrix([[2.2003406982612023e+03, 0., 9.9017948552151142e+02], [0, 2.1989773275557795e+03, 5.6700409473508148e+02], [0., 0., 1.]])
# distortion_coefficients = np.matrix([[-8.2266852133274293e-01], [1.2337344941131776e+00], [-3.4474476305577989e-03], [-5.2138110691419505e-03], [-1.1906641291862861e+00]])

dst = cv2.undistort(img, camera_matrix, distortion_coefficients)

cv2.imwrite("../data/disp/right2_undis.jpg", dst)

# k1 = -0.1335
# k2 = -0.4461
# p1 = -0.00085
# p2 = 0.0020
# k3 = 0.4410
#
# print(img.shape)
#
# img_undis = np.zeros((1944, 2592, 1))
#
# for v in range(img_undis.shape[0]):
#     for u in range(img_undis.shape[1]):
#         x1 = (u-1307.5) / 1853.5
#         y1 = (v-1023.7) / 1853.5
#         r2 = x1 * x1 + y1 * y1
#         x2 = x1 * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * x1 * y1 + p2 * (r2 + 2*x1*x1)
#         y2 = y1 * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + p1 * (r2 + 2*y1*y1) + 2*p2*x1*y1
#         u_dis = x2 * 1853.5 + 1307.5
#         v_dis = y2 * 1853.5 + 1023.7
#         if u_dis >= 0 and v_dis >= 0 and u_dis <= img_undis.shape[1] and v_dis <= img_undis.shape[0]:
#             img_undis[v][u] = img[int(v_dis)][int(u_dis)]
#         else:
#             img_undis[v][u] = 0
#
# cv2.imwrite("data/tsfex.jpg", img_undis)
