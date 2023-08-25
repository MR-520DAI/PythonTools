import cv2
import torch
import numpy as np
import kornia as K
import kornia.feature as KF

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
    # orb_detector = cv2.ORB_create()
    # kp1, des1 = orb_detector.detectAndCompute(img1, None)
    # kp2, des2 = orb_detector.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good_matches.append(m)
    
    
    matches = good_matches

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    img_good_match = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("good match", img_good_match)
    cv2.waitKey(0)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
 
 
    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    return H

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

def LoFTR(img1path, img2path):
    img1 = load_torch_image(img1path)
    img2 = load_torch_image(img2path)
    matcher = KF.LoFTR(pretrained='indoor')
    input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR 只在灰度图上作用
              "image1": K.color.rgb_to_grayscale(img2)}
    with torch.inference_mode():
        correspondences = matcher(input_dict)
    for k,v in correspondences.items():
        print (k)
    
    # 现在让我们用现代RANSAC清理对应关系，并估计两幅图像之间的基本矩阵
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()
    kp0 = []
    kp1 = []
    for i in range(len(confidence)):
        if confidence[i] > 0.9:
            kp0.append([mkpts0[i][0], mkpts0[i][1]])
            kp1.append([mkpts1[i][0], mkpts1[i][1]])
    H, mask = cv2.findHomography(np.array(kp0), np.array(kp1), cv2.RANSAC)
    # img1 = cv2.imread(img1path)
    # img2 = cv2.imread(img2path)
    # kp0 = []
    # num = 0
    # for i in mkpts0:
    #     if confidence[num] > 0.95:
    #         kp0.append(cv2.KeyPoint(i[0], i[1], 1))
    #         num += 1
    #     else:
    #         num += 1
    #         continue
        
    # kp1 = []
    # num = 0
    # for i in mkpts1:
    #     if confidence[num] > 0.95:
    #         kp1.append(cv2.KeyPoint(i[0], i[1], 1))
    #         num += 1
    #     else:
    #         num += 1
    #         continue
    # matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
    # output_img = cv2.drawMatches(img1, kp0, img2, kp1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('Matches', output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return H

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

def Get3DPoints(pixels, img_depth, fx, fy, cx, cy):
    Points = []
    for i in pixels:
        x = int(i[0])
        y = int(i[1])
        Z = img_depth[y][x] / 1250
        X = (x-cx) / fx * Z
        Y = (y-cy) / fy * Z
        Points.append([X, Y, Z])
    return Points

def PixelMatch(pixels1, H):
    pixelsmatch = []
    for i in pixels1:
        x = i[0]
        y = i[1]
        P = np.array([[x],
                      [y],
                      [1.]], dtype=np.float32)
        P = np.matmul(H, P)
        P = P / P[2][0]
        if P[0][0] > 0 and P[0][0] < 639 and P[1][0] > 0 and P[1][0] < 479:
            pixelsmatch.append([x, y, P[0][0], P[1][0]])
        else:
            continue
    return pixelsmatch

def Get3DPointsOnPlane(pixelsmatch, Plane1, Plane2, fx, fy, cx, cy):
    Points = []
    Points1 = []
    Points2 = []
    for i in pixelsmatch:
        x1 = i[0]
        y1 = i[1]
        X1 = (x1-cx) / fx
        Y1 = (y1-cy) / fy
        scale1 = (-Plane1[3]) / (Plane1[0]*X1 + Plane1[1]*Y1 + Plane1[2])
        X1 = X1 * scale1
        Y1 = Y1 * scale1
        Z1 = scale1
        Points1.append([X1, Y1, Z1])
        
        x2 = i[2]
        y2 = i[3]
        X2 = (x2-cx) / fx
        Y2 = (y2-cy) / fy
        scale2 = (-Plane2[3]) / (Plane2[0]*X2 + Plane2[1]*Y2 + Plane2[2])
        X2 = X2 * scale2
        Y2 = Y2 * scale2
        Z2 = scale2
        Points2.append([X2, Y2, Z2])
        Points.append([X1, Y1, Z1, X2, Y2, Z2])
    return Points, Points1, Points2

img1_color = "data\\img\\3.jpg"
img2_color = "data\\img\\4.jpg"
# img1_depth = cv2.imread("data\\img\\1.png", cv2.IMREAD_UNCHANGED)
# img2_depth = cv2.imread("data\\img\\2.png", cv2.IMREAD_UNCHANGED)

# 计算单应矩阵
H = LoFTR(img1_color, img2_color)
# H = CulHomography(cv2.imread(img1_color), cv2.imread(img2_color))
img1 = cv2.imread(img1_color)
img1Reg = cv2.warpPerspective(img1, H, (640, 480))
cv2.imwrite('data\\img\\rst.jpg', img1Reg)

# # 指定像素区域
p1 = (2, 2)
p2 = (220, 3)
p3 = (230, 476)
p4 = (3, 476)
vertices = (p1, p2, p3, p4)
pixels1 = get_polygon_pixels(vertices)
# pixels1 = [(29,31), (132,25), (267,31), (393,27), (23,107), (148,102), (270,106), (395,106),
#            (22,215), (142,221), (254,228), (392,230), (20, 315), (148,309), (271,310), (396,313),
#            (20,398), (146,391), (284,390), (395,380)]
pixelsmatch = PixelMatch(pixels1, H)
# print(len(pixelsmatch))
# pixels1 = [(289,16), (350,19), (416,15), (293,76), (347,67), (412,70), (302,121), (357,119), (410,116), (312,169), (364,168), (422,169), (314,231), (361,232), (417,233), (318, 284),
#            (365,280), (415,278)]
# pixels2 = GetPixelOnImg2(pixels1, H)
# print("pixel1 size:", len(pixels1))
# print("pixel2 size:", len(pixels2))

# 相机参数
fx = 518.9676  # 焦距x（单位像素）
fy = 518.8752  # 焦距y（单位像素）
cx = 320.5550  # 光心x
cy = 237.8842  # 光心y

Plane1 = [0.0621294, -0.0317259, 0.997564, -1.49562]
Plane2 = [0.375814, -0.0428255, 0.925705, -1.50254]
Plane2_1 = [0.927755, 0.00108734, -0.373187, 1.12946]
Plane3 = [0.438069, -0.0324777, 0.898354, -1.50926]
Plane3_1 = [0.89641, 0.00175148, -0.443223, 1.12559]
Plane4 = [-0.799398, -0.0110449, 0.6007, -1.09523]
Plane5 = [0.686264, 0.0157385, -0.727182, 1.10211]
Plane6 = [0.432539, -0.00160202, -0.901614, 1.08332]
Plane7 = [0.198648, -0.00251076, -0.980068, 1.07732]
Plane8 = [-0.045946, 0.0547998, -0.99744, 1.06504]
Plane9 = [-0.350143, 0.0694351, -0.934119, 1.06096]
Plane10 = [-0.476615, 0.0284103, -0.878653, 1.07245]
Plane11 = [-0.701705, 0.0405195, -0.711315, 1.07128]
Plane11_1 = [0.736615, 0.00745304, -0.676272, 1.82056]  # 新平面，存在问题
Plane12 = [0.876555, -0.0181397, 0.480959, -1.18643]
Plane12_1 = [0.526151, 0.020179, -0.850152, 1.78798]

pointsmatch, points1, points2 = Get3DPointsOnPlane(pixelsmatch, Plane3_1, Plane4, fx, fy, cx, cy)
# print(len(pointsmatch))
with open("Points4-3.txt", "w") as f:
    for i in range(0, len(pointsmatch), int(len(pointsmatch)/50)):
        data = "{" + str(pointsmatch[i][0]) + "," + str(pointsmatch[i][1]) + "," + str(pointsmatch[i][2]) + "," + str(pointsmatch[i][3]) + "," + str(pointsmatch[i][4]) + "," + str(pointsmatch[i][5]) + "},\n"
        f.write(data)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.array(points2))
# pcd.voxel_down_sample(voxel_size=0.1)
# o3d.visualization.draw_geometries([pcd])