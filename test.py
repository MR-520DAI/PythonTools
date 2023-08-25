import numpy as np

# def Get3DPointsOnPlane(pixels, Plane, fx, fy, cx, cy):
#     Points = []
#     for i in pixels:
#         x = i[0]
#         y = i[1]
#         X = (x-cx) / fx
#         Y = (y-cy) / fy
#         scale = (-Plane[3]) / (Plane[0]*X + Plane[1]*Y + Plane[2])
#         X = X * scale
#         Y = Y * scale
#         Z = scale
#         Points.append([X, Y, Z])
#     return Points
# def CulDis(p1, p2):
#     for i in range(len(p1)):
#         x1 = p1[i][0]
#         y1 = p1[i][1]
#         z1 = p1[i][2]
#         x2 = p2[i][0]
#         y2 = p2[i][1]
#         z2 = p2[i][2]
#         print(np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))

# # 相机参数
# fx = 518.9676  # 焦距x（单位像素）
# fy = 518.8752  # 焦距y（单位像素）
# cx = 320.5550  # 光心x
# cy = 237.8842  # 光心y

# Plane2 = [0.375814, -0.0428255, 0.925705, -1.50254]
# Plane2_1 = [0.927755, 0.00108734, -0.373187, 1.12946]

# # pixels = [[178,6]]
# # p1 = Get3DPointsOnPlane(pixels, Plane2, fx, fy, cx, cy)
# # print(p1)
# # p2 = Get3DPointsOnPlane(pixels, Plane2_1, fx, fy, cx, cy)
# # print(p2)

# Plane12 = [0.876555, -0.0181397, 0.480959, -1.18643]
# Plane12_1 = [-0.528174890260473, -0.016417365425806688, 0.8489768874420626, -1.7840523482920017]
# pixels = [[341, 3]]
# p1 = Get3DPointsOnPlane(pixels, Plane12, fx, fy, cx, cy)
# print("p1:",p1)
# p2 = Get3DPointsOnPlane(pixels, Plane12_1, fx, fy, cx, cy)
# print("p2:",p2)
# CulDis(p1, p2)

Vertices = [[(6, 6), (600, 6), (600, 460), (6, 460)],
              [[(178, 3), (636, 3), (636, 476), (189, 476)], [(3, 3), (179, 3), (189, 476), (3, 476)]],
              [(2, 2), (220, 3), (230, 476), (3, 476)],
              [(2, 2), (328, 2), (337, 476), (3, 476)],
              [(2, 2), (418, 2), (424, 476), (3, 476)],
              [(2, 2), (500, 2), (500, 476), (3, 476)],
              [(2, 2), (625, 2), (625, 476), (3, 476)],
              [(2, 2), (625, 2), (625, 476), (3, 476)],
              [(2, 2), (500, 2), (500, 476), (3, 476)],
              [(20, 2), (500, 2), (500, 476), (20, 476)]]
print(Vertices[0][0])