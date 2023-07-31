import numpy as np

p = np.matrix([[0.6756324958534569], [-0.23967176169299736], [0.2542785444222126]], dtype="float")

# 定义相机的内参数矩阵（示例参数）
fx = 518.9676  # 焦距
fy = 518.8752
cx = 320.555  # 图像中心点坐标
cy = 237.8842
cameraMatrix = np.matrix([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype="float")

rotation = np.matrix([
[-0.01256177,  0.00932273, -0.99987764],
 [ 0.00134036, -0.99995548, -0.00934029],
 [-0.9999202,  -0.00145753,  0.01254872]], dtype="float")
translation = np.matrix([[-0.00595316], [-0.08529437], [-0.00214798]], dtype="float")

pcam = rotation * p + translation
pimg = cameraMatrix * pcam

print(pimg[0] / pimg[2])
print(pimg[1] / pimg[2])