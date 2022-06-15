import numpy as np
import cv2
import os
#双目相机参数
class stereoCameral(object):
    def __init__(self):

        #左相机内参数
        self.cam_matrix_left = np.array([[5.3340331777463712e+02, 0., 3.4251343227755160e+02],
                               [0., 5.3343402398745684e+02, 2.3475353096292952e+02],
                               [0., 0., 1.]])
        #右相机内参数
        self.cam_matrix_right = np.array([[5.3699964365956180e+02, 0., 3.2744774682047540e+02],
                                [0., 5.3658501956219982e+02, 2.4990007115682096e+02],
                                [0., 0., 1.]])

        #左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-2.8214652038759541e-01, 4.0840748605552028e-02,
                               1.2058218262263004e-03, -1.2307876204068898e-04,
                               1.1409651538056684e-01]])
        self.distortion_r = np.array([[-2.9611763213840986e-01, 1.3891105660442912e-01,
                               -5.0433529470851200e-04, 9.4658617944131683e-05,
                               -4.9061152399050519e-02]])

        #旋转矩阵
        self.R = np.array([[9.9998525681254691e-01, 3.5314167630729580e-03, 4.1249549318989115e-03],
                              [-3.5020197622120568e-03, 9.9996857547469531e-01, -7.1122373901368894e-03],
                              [-4.1499415814907617e-03, 7.0976868593980872e-03, 9.9996619984183277e-01]])
        #平移矩阵
        self.T = np.array([-6.6541611689169670e+01, 7.3831486397326695e-01, -9.7009228998274979e-02])

def getRectifyTransform(height, width, config):
    #读取矩阵参数
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    #计算校正变换
    if type(height) != "int" or type(width) != "int":
        height = int(height)
        width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q

# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2

#视差计算
def sgbm(imgL, imgR):
    #SGBM参数设置
    blockSize = 5
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity = 1,
                                   numDisparities = 64,
                                   blockSize = blockSize,
                                   P1 = 8 * img_channels * blockSize * blockSize,
                                   P2 = 32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff = -1,
                                   preFilterCap = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 100,
                                   mode = cv2.STEREO_SGBM_MODE_HH)
    # 计算视差图
    disp = stereo.compute(imgL, imgR)
    disp = np.divide(disp.astype(np.float32), 16.)#除以16得到真实视差图
    return disp
#计算三维坐标，并删除错误点
def threeD(disp, Q):
    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)

    points_3d = points_3d.reshape(points_3d.shape[0] * points_3d.shape[1], 3)

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    #选择并删除错误的点
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    points_3d = np.delete(points_3d, remove_idx, 0)

    #计算目标点（这里我选择的是目标区域的中位数，可根据实际情况选取）
    if points_3d.any():
        x = np.median(points_3d[:, 0])
        y = np.median(points_3d[:, 1])
        z = np.median(points_3d[:, 2])
        targetPoint = [x, y, z]
    else:
        targetPoint = [0, 0, -1]#无法识别目标区域

    return targetPoint

# fx = 5.3340331777463712e+02
# fy = 5.3343402398745684e+02
# cx = 3.4251343227755160e+02
# cy = 2.3475353096292952e+02
# baseline = 66

# name = os.listdir("Y:\\data\\kitti\\stereo\\2015\\training\\image_2\\")
# for i in name:
#     imgL = cv2.imread("Y:\\data\\kitti\\stereo\\2015\\training\\image_2\\" + i)
#     imgR = cv2.imread("Y:\\data\\kitti\\stereo\\2015\\training\\image_3\\" + i)
#     disp = sgbm(imgL, imgR)
#     cv2.imwrite("kitti_disp\\" + i, disp)
imgL = cv2.imread("l.jpg")
imgR = cv2.imread("r.jpg")

# height, width = imgL.shape[0:2]
# 读取相机内参和外参
# config = stereoCameral()
#
# map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
# iml_rectified, imr_rectified = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)
# cv2.imwrite("iml_rectified.jpg", iml_rectified)
# cv2.imwrite("imr_rectified.jpg", imr_rectified)

disp = sgbm(imgL, imgR)

# disp = np.uint16(disp)

cv2.imwrite("disp.png", disp)
# target_point = threeD(disp, Q)#计算目标点的3D坐标（左相机坐标系下）
# print(target_point)

# depth = np.zeros(shape=(480, 640), dtype=np.uint16)
# for i in range(disp.shape[0]):
#     for j in range(disp.shape[1]):
#         if (disp[i][j]) <= 2.5:
#             continue
#         else:
#             depth[i][j] = np.int16(fx * baseline / disp[i][j])
#
# cv2.imwrite("depth.png", depth)
