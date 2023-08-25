import cv2
import numpy as np
import open3d as o3d

def get_polygon_pixels(vertices, scale=2):
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

def Get3DPointsOnPlane(pixels, Plane, img, fx, fy, cx, cy):
    Points = []
    Colors = []
    for i in pixels:
        x = i[0]
        y = i[1]
        X = (x-cx) / fx
        Y = (y-cy) / fy
        scale = (-Plane[3]) / (Plane[0]*X + Plane[1]*Y + Plane[2])
        X = X * scale
        Y = Y * scale
        Z = scale
        Points.append([X, Y, Z])
        Colors.append([img[y][x][2]/255, img[y][x][1]/255, img[y][x][0]/255])
    return Points, Colors

def GetPcd(img, pixels, plane, fx, fy, cx, cy, T):
    points, colors = Get3DPointsOnPlane(pixels, plane, img, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(T)

    return pcd

if __name__ == "__main__":
    # 相机参数
    fx = 518.9676  # 焦距x（单位像素）
    fy = 518.8752  # 焦距y（单位像素）
    cx = 320.5550  # 光心x
    cy = 237.8842  # 光心y
    
    p1 = (6, 6)
    p2 = (600, 6)
    p3 = (600, 460)
    p4 = (6, 460)
    vertices1 = (p1, p2, p3, p4)
    pixels1 = get_polygon_pixels(vertices1)
    img1 = cv2.imread("data\\img\\1.jpg")
    plane1 = [0.0621294, -0.0317259, 0.997564, -1.49562]
    T1 = np.array([[1.,0.,0.,0.],
                   [0.,1.,0.,0.],
                   [0.,0.,1.,0.],
                   [0.,0.,0.,1.]])
    pcd1 = GetPcd(img1, pixels1, plane1, fx, fy, cx, cy, T1)

    p1 = (178, 3)
    p2 = (636, 3)
    p3 = (636, 476)
    p4 = (189, 476)
    vertices2 = (p1, p2, p3, p4)
    img2 = cv2.imread("data\\img\\2.jpg")
    plane2 = [0.375814, -0.0428255, 0.925705, -1.50254]
    pixels2 = get_polygon_pixels(vertices2)
    p1 = (3, 3)
    p2 = (179, 3)
    p3 = (189, 476)
    p4 = (3, 476)
    vertices2_1 = (p1, p2, p3, p4)
    plane2_1 = [0.927755, 0.00108734, -0.373187, 1.12946]
    pixels2_1 = get_polygon_pixels(vertices2_1)
    T2 = np.array([[0.948173,0.00153341,-0.31775,-0.00973944],
                  [0.00211253,0.999936,0.0111294,-0.00935452],
                  [0.317746,-0.0112238,0.948109,-0.006628],
                  [0.,0.,0.,1.]])
    pcd2 = GetPcd(img2, pixels2, plane2, fx, fy, cx, cy, T2)
    pcd2_1 = GetPcd(img2, pixels2_1, plane2_1, fx, fy, cx, cy, T2)

    p1 = (2, 2)
    p2 = (220, 3)
    p3 = (230, 476)
    p4 = (3, 476)
    vertices3 = (p1, p2, p3, p4)
    img3 = cv2.imread("data\\img\\3.jpg")
    plane3 = [0.89641, 0.00175148, -0.443223, 1.12559]
    pixels3 = get_polygon_pixels(vertices3)
    T3 = np.array([[0.997677,0.00194522,-0.0680957,-0.0145153],
                  [-0.00264787,0.999944,-0.0102298,0.0173881],
                  [0.068072,0.0103863,0.997626,-0.000563012],
                  [0.,0.,0.,1.]])
    T3 = np.matmul(T2, T3)
    pcd3 = GetPcd(img3, pixels3, plane3, fx, fy, cx, cy, T3)

    p1 = (2, 2)
    p2 = (328, 2)
    p3 = (337, 476)
    p4 = (3, 476)
    vertices4 = (p1, p2, p3, p4)
    img4 = cv2.imread("data\\img\\4.jpg")
    plane4 = [-0.799398, -0.0110449, 0.6007, -1.09523]
    pixels4 = get_polygon_pixels(vertices4)
    T4 = np.array([[0.982871,0.00760603,-0.184138,-0.0302192],
                  [-0.00644794,0.999955,0.00688719,0.0129755],
                  [0.184182,-0.00558191,0.982876,0.00747371],
                  [0.,0.,0.,1.]])
    T4 = np.matmul(T3, T4)
    pcd4 = GetPcd(img4, pixels4, plane4, fx, fy, cx, cy, T4)

    p1 = (2, 2)
    p2 = (418, 2)
    p3 = (424, 476)
    p4 = (3, 476)
    vertices5 = (p1, p2, p3, p4)
    img5 = cv2.imread("data\\img\\5.jpg")
    plane5 = [0.686264, 0.0157385, -0.727182, 1.10211]
    pixels5 = get_polygon_pixels(vertices5)
    T5 = np.array([[0.985587,0.00675817,-0.169034,0.00485427],
                  [-0.00686024,0.999976,0.0,-0.0125871],
                  [0.16903,0.00117914,0.98561,-0.00522477],
                  [0.,0.,0.,1.]])
    T5 = np.matmul(T4, T5)
    pcd5 = GetPcd(img5, pixels5, plane5, fx, fy, cx, cy, T5)

    p1 = (2, 2)
    p2 = (500, 2)
    p3 = (500, 476)
    p4 = (3, 476)
    vertices6 = (p1, p2, p3, p4)
    img6 = cv2.imread("data\\img\\6.jpg")
    plane6 = [0.432539, -0.00160202, -0.901614, 1.08332]
    pixels6 = get_polygon_pixels(vertices6)
    T6 = np.array([[0.952587,-0.00863646,-0.304145,-0.0186172],
                  [0.00345216,0.99984,-0.0175791,0.041585],
                  [0.304248,0.0156957,0.952464,0.00916842],
                  [0.,0.,0.,1.]])
    T6 = np.matmul(T5, T6)
    pcd6 = GetPcd(img6, pixels6, plane6, fx, fy, cx, cy, T6)

    p1 = (2, 2)
    p2 = (625, 2)
    p3 = (625, 476)
    p4 = (3, 476)
    vertices7 = (p1, p2, p3, p4)
    img7 = cv2.imread("data\\img\\7.jpg")
    plane7 = [0.198648, -0.00251076, -0.980068, 1.07732]
    pixels7 = get_polygon_pixels(vertices7)
    T7 = np.array([[0.96954,0.00674464,-0.244838,-0.00852559],
                  [-0.00757853,0.999968,-0.00246393,0.0132022],
                  [0.244814,0.0042444,0.969561,0.00254155],
                  [0.,0.,0.,1.]])
    T7 = np.matmul(T6, T7)
    pcd7 = GetPcd(img7, pixels7, plane7, fx, fy, cx, cy, T7)

    p1 = (2, 2)
    p2 = (625, 2)
    p3 = (625, 476)
    p4 = (3, 476)
    vertices8 = (p1, p2, p3, p4)
    img8 = cv2.imread("data\\img\\8.jpg")
    plane8 = [-0.045946, 0.0547998, -0.99744, 1.06504]
    pixels8 = get_polygon_pixels(vertices8)
    # pixels8 = [(176,259), (611,251)]
    T8 = np.array([[0.969901,0.00952862,-0.243312,-0.0308529],
                  [0.00453076,0.998355,0.0571585,-0.0589847],
                  [0.243456,-0.0565405,0.968263,0.00642734],
                  [0.,0.,0.,1.]])
    T8 = np.matmul(T7, T8)
    pcd8 = GetPcd(img8, pixels8, plane8, fx, fy, cx, cy, T8)

    p1 = (2, 2)
    p2 = (500, 2)
    p3 = (500, 476)
    p4 = (3, 476)
    vertices9 = (p1, p2, p3, p4)
    img9 = cv2.imread("data\\img\\9.jpg")
    plane9 = [-0.350143, 0.0694351, -0.934119, 1.06096]
    pixels9 = get_polygon_pixels(vertices9)
    T9 = np.array([[0.951511,-0.00180346,-0.30761,0.00731651],
                  [0.00620559,0.999892,0.0133332,-0.00976963],
                  [0.307553,-0.0145956,0.951419,0.00321626],
                  [0.,0.,0.,1.]])
    T9 = np.matmul(T8, T9)
    pcd9 = GetPcd(img9, pixels9, plane9, fx, fy, cx, cy, T9)

    p1 = (20, 2)
    p2 = (500, 2)
    p3 = (500, 476)
    p4 = (20, 476)
    vertices10 = (p1, p2, p3, p4)
    img10 = cv2.imread("data\\img\\10.jpg")
    plane10 = [-0.476615, 0.0284103, -0.878653, 1.07245]
    pixels10 = get_polygon_pixels(vertices10)
    T10 = np.array([[0.99033,0.0100119,-0.138367,0.00815628],
                  [-0.0154566,0.999146,-0.0383313,0.0538661],
                  [0.137865,0.0400993,0.989639,-0.0113552],
                  [0.,0.,0.,1.]])
    T10 = np.matmul(T9, T10)
    pcd10 = GetPcd(img10, pixels10, plane10, fx, fy, cx, cy, T10)

    p1 = (161, 2)
    p2 = (500, 2)
    p3 = (500, 476)
    p4 = (175, 476)
    vertices11 = (p1, p2, p3, p4)
    img11 = cv2.imread("data\\img\\11.jpg")
    plane11 = [-0.701705, 0.0405195, -0.711315, 1.07128]
    pixels11 = get_polygon_pixels(vertices11)
    # p1 = (2, 2)
    # p2 = (160, 2)
    # p3 = (174, 476)
    # p4 = (2, 476)
    # vertices11_1 = (p1, p2, p3, p4)
    # plane11_1 = [0.739183, 0.00762266, -0.673462, 1.80394]
    # pixels11_1 = get_polygon_pixels(vertices11_1)
    T11 = np.array([[0.960723,0.00365811,-0.277486,-0.000701992],
                  [0.000862231,0.999869,0.0161665,-0.0134937],
                  [0.277509,-0.0157708,0.960594,-0.00517578],
                  [0.,0.,0.,1.]])
    T11 = np.matmul(T10, T11)
    pcd11 = GetPcd(img11, pixels11, plane11, fx, fy, cx, cy, T11)
    # pcd11_1 = GetPcd(img11, pixels11_1, plane11_1, fx, fy, cx, cy, T11)

    p1 = (290, 2)
    p2 = (630, 2)
    p3 = (630, 476)
    p4 = (300, 476)
    vertices12 = (p1, p2, p3, p4)
    img12 = cv2.imread("data\\img\\test.jpg")
    # plane12 = [0.876555, -0.0181397, 0.480959, -1.18643]
    plane12 = [-0.815081, 0.0357184, -0.578245, 1.13176]
    pixels12 = get_polygon_pixels(vertices12)
    p1 = (2, 2)
    p2 = (289, 2)
    p3 = (299, 476)
    p4 = (2, 476)
    vertices12_1 = (p1, p2, p3, p4)
    # plane12_1 = [0.526151, 0.020179, -0.850152, 1.78798]
    plane12_1 = [0.578417, 0.00710349, -0.81571, 1.75389]
    pixels12_1 = get_polygon_pixels(vertices12_1)
    T12 = np.array([[0.95797,0.0195918,-0.286199,-0.0835266],
                  [-0.0222279,0.999735,-0.00596444,0.00645138],
                  [0.286006,0.0120754,0.958152,-0.0758167],
                  [0.,0.,0.,1.]])
    T12 = np.matmul(T11, T12)
    T12 = np.array([[1.,0.,0.,0.],
                   [0.,1.,0.,0.],
                   [0.,0.,1.,0.],
                   [0.,0.,0.,1.]])
    pcd12 = GetPcd(img12, pixels12, plane12, fx, fy, cx, cy, T12)
    pcd12_1 = GetPcd(img12, pixels12_1, plane12_1, fx, fy, cx, cy, T12)

    pcd_all = o3d.geometry.PointCloud()
    # pcd_all += pcd1
    # pcd_all += pcd2
    # pcd_all += pcd2_1
    # pcd_all += pcd3
    # pcd_all += pcd4
    # pcd_all += pcd5
    # pcd_all += pcd6
    # pcd_all += pcd7
    # pcd_all += pcd8
    # pcd_all += pcd9
    # pcd_all += pcd10
    # pcd_all += pcd11
    pcd_all += pcd12
    pcd_all += pcd12_1
    pcd_all.voxel_down_sample(voxel_size=0.1)
    o3d.visualization.draw_geometries([pcd_all])
    o3d.io.write_point_cloud("output.ply", pcd_all)