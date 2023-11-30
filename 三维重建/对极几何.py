import cv2
import time
import torch
import numpy as np
import kornia as K
import kornia.feature as KF

def Get3DPointsOnPlane(pixel, plane, fx, cx, cy):
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
    Y = (y-cy) / fx
    scale = (-plane[3]) / (plane[0]*X + plane[1]*Y + plane[2])
    X = X * scale
    Y = Y * scale
    Z = scale
    return [X, Y, Z]

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2

def DISKFeatureMatch(img1path, img2path):
    # device = K.utils.get_cuda_or_mps_device_if_available()
    img1 = K.io.load_image(img1path, K.io.ImageLoadType.RGB32)[None, ...]
    img2 = K.io.load_image(img2path, K.io.ImageLoadType.RGB32)[None, ...]
    num_features = 2048
    disk = KF.DISK.from_pretrained("depth")

    # hw1 = torch.tensor(img1.shape[2:], device=device)
    # hw2 = torch.tensor(img2.shape[2:], device=device)
    with torch.inference_mode():
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = disk(inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors
        dists, idxs = KF.match_smnn(descs1, descs2, 0.98)
        mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)
    mkpts1 = mkpts1.detach().numpy()
    mkpts2 = mkpts2.detach().numpy()
    # if mkpts1.shape[0] < 200:
    #     E = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    #     return E
    # H, mask = cv2.findHomography(mkpts1.detach().numpy(), mkpts2.detach().numpy(), cv2.RANSAC)

    # E, mask = cv2.findEssentialMat(mkpts1, mkpts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    # _, R, t, mask = cv2.recoverPose(E, mkpts1, mkpts2, focal=fx, pp=(cx, cy))
    return mkpts1, mkpts2

def GetPose(mkpts1, mkpts2, fx, cx, cy):
    if mkpts1.shape[0] < 200:
        R = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        t = np.array([[0.], [0.], [0.]])
        return R, t
    E, mask = cv2.findEssentialMat(mkpts1, mkpts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, mkpts1, mkpts2, focal=fx, pp=(cx, cy), mask=mask)
    return R, t, mask

def pixel2cam(mkpts, fx, cx, cy):
    mkpoints = []
    for pixel in mkpts:
        x = (pixel[0] - cx) / fx
        y = (pixel[1] - cy) / fx
        mkpoints.append([x, y])
    return np.array(mkpoints)

def TriangulatePoints(mkpts1, mkpts2, R, t, fx, cx, cy):
    mkpoints1 = pixel2cam(mkpts1, fx, cx, cy)
    mkpoints2 = pixel2cam(mkpts2, fx, cx, cy)
    T1 = np.array([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.]])
    T2 = np.array([[R[0][0], R[0][1], R[0][2], t[0][0]],
                   [R[1][0], R[1][1], R[1][2], t[1][0]],
                   [R[2][0], R[2][1], R[2][2], t[2][0]]])
    points4D = cv2.triangulatePoints(T1, T2, mkpoints1.T, mkpoints2.T)
    points3D = []
    for p4d in points4D.T:
        X = p4d[0] / p4d[3]
        Y = p4d[1] / p4d[3]
        Z = p4d[2] / p4d[3]
        points3D.append([X, Y, Z])
    return points3D

def IfPointOnPlane(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    z1 = point1[2]
    x2 = point2[0]
    y2 = point2[1]
    z2 = point2[2]
    scalex = x2 / x1
    scaley = y2 / y1
    scalez = z2 / z1
    thready = scaley / scalex
    threadz = scalez / scalex
    if thready > 0.98 and thready < 1.02 and threadz > 0.98 and threadz < 1.02:
        return True
    return False

def tmp(point1, point2, thread):
    x1 = point1[0]
    y1 = point1[1]
    z1 = point1[2]
    x2 = point2[0]
    y2 = point2[1]
    z2 = point2[2]
    scalex = x2 / x1
    scaley = y2 / y1
    scalez = z2 / z1
    threadx = scalex / thread
    thready = scaley / thread
    threadz = scalez / thread
    if threadx > 0.9 and threadx < 1.1 and thready > 0.9 and thready < 1.1 and threadz > 0.9 and threadz < 1.1:
        return True
    return False

def DrawPoints(img1_path, img2_path, mkpts1, mkpts2):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    for p1, p2 in zip(mkpts1, mkpts2):
        cv2.circle(img1, (int(p1[0]), int(p1[1])), 3, (0,255,0), 2)
        cv2.circle(img2, (int(p2[0]), int(p2[1])), 3, (0,255,0), 2)
    cv2.imshow("p1", img1)
    cv2.imshow("pw", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("img1.jpg", img1)
    # cv2.imwrite("img2.jpg", img2)

if __name__ == "__main__":
    # 相机参数
    fx = 406.1814  # 焦距x（单位像素）
    fy = 406.1814  # 焦距y（单位像素）
    cx = 311.5565  # 光心x
    cy = 250.0853  # 光心y
    Kmat = np.array([[fx, 0., cx], 
                  [0., fy, cy], 
                  [0., 0., 1.]])
    dist_coeffs = np.zeros((5,1))

    img1_path = "E:\\data\\MYData\\CamLidardata\\2.jpg"
    img2_path = "E:\\data\\MYData\\CamLidardata\\3.jpg"
    mkpts1, mkpts2 = DISKFeatureMatch(img1_path, img2_path)

    DrawPoints(img1_path, img2_path, mkpts1, mkpts2)

    R, t, mask = GetPose(mkpts1, mkpts2, fx, cx, cy)
    print("R:\n", R)
    print("t:\n", t)
    mask = mask.flatten()
    mkpts1 = mkpts1[mask == 1]
    mkpts2 = mkpts2[mask == 1]
    points = TriangulatePoints(mkpts1, mkpts2, R, t, fx, cx, cy)
    with open("data.txt", "w") as f:
        for point, pixel in zip(points, mkpts1):
            data = str(point[0]) + "," + str(point[1]) + "," + str(point[2]) + ": " + str(pixel[0]) + "," + str(pixel[1]) + "\n" 
            f.write(data)

    planes1 = [[-0.7123993451943393, 0.012967548961950152, -0.7016544845154171, 1.6062702755554301],
               [0.7004411136658442, 0.058467531581345926, -0.7113113200542617, 1.7399609352710308]]
    vertices = [[[299.9106344553329, 190.409226651851], [558.3510898472555, 172.93321897612438], [551.8286547351394, 321.41830897061885], [297.20950229058695, 279.8236751389669]],
                [[56.930624055125264, 161.49484097059315], [299.8925085827945, 191.0092398305916], [296.5783765822489, 300.715565537228], [60.93160976244127, 332.70074203787397]]]
    

    plane1L = [0.7027575706848495, 0.06359998818984774, -0.708580862250301, 1.7351590024687176]
    plane1R = [-0.7095214122572147, 0.009204836629550697, -0.7046237552986353, 1.610648716984899]

    plane2L = [0.5437833152069513,0.04477061508915761,-0.838030606915449,1.7050009520395548]
    plane2R = [-0.8348412147388603,0.006905511143266526,-0.5504475089317732,1.6460371223983763]

    points1_select = []
    points2_select = []
    pixels1_select = []
    pixels2_select = []
    for pixel1, pixel2, point1 in zip(mkpts1, mkpts2, points):
        point2 = Get3DPointsOnPlane(pixel2, plane2L, fx, cx, cy)
        p = np.array([[point1[0]], [point1[1]], [point1[2]]])
        p = np.matmul(R, p)
        p = p + t
        point1 = [p[0][0], p[1][0], p[2][0]]
        if IfPointOnPlane(point2, point1):
            # if tmp(point2, point1, 10):
            pixels1_select.append([pixel1[0], pixel1[1]])
            pixels2_select.append([pixel2[0], pixel2[1]])
            points2_select.append([point2[0], point2[1], point2[2]])
            p = Get3DPointsOnPlane(pixel1, plane1L, fx, cx, cy)
            points1_select.append(p)
        else:
            continue
    print(pixels1_select)
    with open("data1.txt", "w") as f:
        for point, pixel in zip(points1_select, points2_select):
            data = str(point[0]) + "," + str(point[1]) + "," + str(point[2]) + "," + str(pixel[0]) + "," + str(pixel[1]) + "," + str(pixel[2]) + "\n"
            f.write(data)

    # 计算位姿
    # dist_coeffs = np.zeros((4, 1))
    # retval, rvec, tvec = cv2.solvePnP(np.array(points2_select), np.array(pixels1_select), Kmat, dist_coeffs)
    # rotation_matrix, _ = cv2.Rodrigues(rvec)
    # print(rotation_matrix)
    # print(tvec)
