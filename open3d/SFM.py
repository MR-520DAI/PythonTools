import cv2
import numpy as np
from scipy.optimize import curve_fit

def CulHomography(img1, img2):
    """通过SIFT特征点匹配计算单应矩阵H

    Args:
        img1 (cv2 Mat): 读入图像数据
        img2 (cv2 Mat): 读入图像数据

    Returns:
        cv2 Mat: 单应矩阵H
    """
    sift_detector = cv2.SIFT_create()
    kp1, des1 = sift_detector.detectAndCompute(img1, None)
    kp2, des2 = sift_detector.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
    
    
    matches = good_matches

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
 
 
    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    return H

def plane(xy, a, b, c, d):
    """定义平面方程公式

    Args:
        xy (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_
        c (_type_): _description_
        d (_type_): _description_

    Returns:
        _type_: _description_
    """
    x, y = xy
    return a/(-c)*x + b/(-c)*y + d/(-c)

def CulPlane(points):
    """计算平面方程参数

    Args:
        points (array): 三维坐标集合

    Returns:
        float: 平面方程的参数a,b,c,d
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    parameters, _ = curve_fit(plane, (x, y), z)
    # 提取平面参数
    a, b, c, d = parameters

    # 归一化
    scale = np.sqrt(a*a + b*b + c*c)
    return [a/scale, b/scale, c/scale, d/scale]

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

def CulTransformation(Points3D, Points2D, camera_matrix, dist_coeffs):
    """计算位姿

    Args:
        Points3D (_type_): 三维坐标点集
        Points2D (_type_): 二维坐标点集
        camera_matrix (_type_): 相机内参
        dist_coeffs (_type_): 畸变系数

    Returns:
        mat: 变换矩阵
    """
    retval, rvec, tvec = cv2.solvePnP(Points3D, Points2D, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    homogeneous_matrix = np.column_stack((rotation_matrix, tvec))
    homogeneous_matrix = np.vstack((homogeneous_matrix, [0, 0, 0, 1]))
    return homogeneous_matrix

def PixelMatch(pixels1, H, width, height):
    pixelmatch = []
    for i in pixels1:
        x = i[0]
        y = i[1]
        P = np.array([[x],
                      [y],
                      [1.]], dtype=np.float32)
        P = np.matmul(H, P)
        P = P / P[2][0]
        if P[0][0] > 0 and P[0][0] < (width-1) and P[1][0] > 0 and P[1][0] < (height-1):
            pixelmatch.append(([x, y], [P[0][0], P[1][0]]))
        else:
            continue
    return pixelmatch

def PointsMatch(pixelsmatch, plane2, fx, fy, cx, cy):
    points2D = []
    points3D = []
    for i in pixelsmatch:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[1][0]
        y2 = i[1][1]
        X = (x2-cx) / fx
        Y = (y2-cy) / fy
        scale = -plane2[3] / (plane2[0]*X+plane2[1]*Y+plane2[2])
        X = X * scale
        Y = Y * scale
        Z = scale
        points3D.append([X, Y, Z])
        points2D.append([x1, y1])
    return points2D, points3D

if __name__ == "__main__":
    # 相机参数
    fx = 518.9676  # 焦距x（单位像素）
    fy = 518.8752  # 焦距y（单位像素）
    cx = 320.5550  # 光心x
    cy = 237.8842  # 光心y
    camera_matrix = np.array([[518.9676, 0, 320.5550], [0, 518.8752, 237.8842], [0, 0, 1.]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    img1 = cv2.imread("data\\img\\1.jpg")
    img2 = cv2.imread("data\\img\\2.jpg")
    # 计算单应矩阵
    H = CulHomography(img1, img2)
    print("单应矩阵H:\n",H)
    # img1Reg = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    # cv2.imwrite('rst.jpg', img1Reg)

    # 指定像素区域
    p1 = (284, 17)
    p2 = (416, 21)
    p3 = (430, 287)
    p4 = (303, 293)
    vertices = (p1, p2, p3, p4)
    pixels1 = get_polygon_pixels(vertices)
    pixelsmatch = PixelMatch(pixels1, H, 640, 480)

    lidarpoints1 = np.array([[-0.17930150775210185,-1.261012716599863,2.948872544905393],
                            [0.1669407250305636,-1.2412020417742566,2.942327302135219],
                            [0.539215661551193,-1.2594064977606867,2.931902747735711],
                            [-0.15805852402520004,-0.9287510149053656,2.976855484409377],
                            [0.1511072430377644,-0.9766095152038363,2.9653909344649394],
                            [0.5212595387091153,-0.9571526538550011,2.958246069013906],
                            [-0.10716747662292313,-0.6752042793207,2.9973833544087576],
                            [0.20990596327044006,-0.6848386981515576,2.9890079293222236],
                            [0.5136885625841091,-0.7001136902531161,2.9804653191539483],
                            [-0.049776052959755734,-0.4008643105213392,3.019539303564852],
                            [0.2521423593553042,-0.4056601434424947,3.0119395809174767],
                            [0.5873116012505066,-0.3988732238508117,3.004541299749938],
                            [-0.03852625962077757,-0.040468300021009235,3.050172462604397],
                            [0.23723888362655415,-0.034521192441377836,3.0441165548856994],
                            [0.5643660960466407,-0.02858590650455774,3.0368367296043814],
                            [-0.015146483983743382,0.2734311592364072,3.076530114082865],
                            [0.2627317241014442,0.24900724944078187,3.067819829019882],
                            [0.5568326314340725,0.23655846612654813,3.059750059156389]])
    lidarpoints2 = np.array([[-1.0854564602962404,-1.1758570071397338,2.758627105625414],
                            [-0.767788305482848,-1.17727938306698,2.888662736958905],
                            [-0.4177042710551994,-1.2194438730182542,3.0308544172896243],
                            [-1.0628647976019063,-0.8451332522143491,2.7769463841934092],
                            [-0.7805185032337765,-0.908876233081389,2.8908099923609996],
                            [-0.4335769852700717,-0.90868362303702,3.032876384944284],
                            [-1.0143298529108473,-0.5952589858082807,2.80367161681424],
                            [-0.7240300508156787,-0.618045706478675,2.9219149809785616],
                            [-0.4397004668326543,-0.6465057232066599,3.0375581709045645],
                            [-0.9597750387631979,-0.3253294266276228,2.8334117213229617],
                            [-0.68305771917355,-0.3397188292956985,2.9463237686715225],
                            [-0.3687990179013866,-0.34379341704053384,3.074890615216065],
                            [-0.9463690482627961,0.029985137824995615,2.8486440717608885],
                            [-0.6944343788997408,0.030611807514061447,2.951820195037031],
                            [-0.38920818766086446,0.030246567126978825,3.076790240157232],
                            [-0.9224934995453242,0.33758957430899234,2.8668551071732753],
                            [-0.6691252516199748,0.3111449528991072,2.9698759148692484],
                            [-0.39534350558129705,0.2959646815215394,3.081564255331599]])
    plane1 = CulPlane(lidarpoints1)
    plane2 = CulPlane(lidarpoints2)
    print("plane1:",plane1)
    print("plane2:",plane2)

    # points2D, points3D = PointsMatch(pixelsmatch, plane2, fx, fy, cx, cy)
    # points2D_select = []
    # points3D_select = []
    # for i in range(0, len(points2D), int(len(points2D)/100)):
    #     points2D_select.append(points2D[i])
    #     points3D_select.append(points3D[i])
    # # print(points2D_select)
    # p = np.array(points2D_select, dtype=np.float64)
    # T = CulTransformation(np.array(points3D_select, dtype=np.float64), np.array(points2D_select, dtype=np.float64), camera_matrix, dist_coeffs)
    # print("变换矩阵T:\n",T)
