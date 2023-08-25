import numpy as np
import matplotlib.pyplot as plt        
from mpl_toolkits.mplot3d import Axes3D

# 上、左、前、右、下、后，6个平面方程
# 0   1   2   3   4   5
plane_set = [[-0.999218 ,-0.00701429 ,-0.038925 ,1.1937],
            
             [0.018447 ,0.999676 ,-0.017548 ,1.63135],

             [-0.0538746 ,0.012239 ,0.998473 ,-0.922939],
             
             [0.0199784 ,0.998334 ,-0.0541366 ,-1.1185],
             
             [-0.998383 ,0.029792 ,-0.0484061 ,-1.5029],
             [-0.0328839 ,0.0114375 ,0.999394 ,2.46233]]

'''
左上上、左上下、右上上、右上下
左下上、左下下、右下上、右下下
'''
index_id_set = [[0,1,2],
                [4,1,2],
                [0,2,3],
                [4,2,3],
                [0,1,5],
                [4,1,5],
                [0,3,5],
                [4,3,5]]

point_set = []
for index_id in index_id_set:
    A = np.array([[plane_set[index_id[0]][0], plane_set[index_id[0]][1], plane_set[index_id[0]][2]],
              [plane_set[index_id[1]][0], plane_set[index_id[1]][1], plane_set[index_id[1]][2]],
              [plane_set[index_id[2]][0], plane_set[index_id[2]][1], plane_set[index_id[2]][2]]])
    b = np.array([-plane_set[index_id[0]][3], -plane_set[index_id[1]][3], -plane_set[index_id[2]][3]])
    p = np.linalg.solve(A, b)
    point_set.append(p.tolist())

points = np.array(point_set)
print(points)

# 画出立方体的三维模型
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# 画出立方体的12条边
ax.plot([points[0, 0], points[1, 0]], [points[0, 1], points[1, 1]], [points[0, 2], points[1, 2]])
ax.plot([points[0, 0], points[2, 0]], [points[0, 1], points[2, 1]], [points[0, 2], points[2, 2]])
ax.plot([points[0, 0], points[4, 0]], [points[0, 1], points[4, 1]], [points[0, 2], points[4, 2]])
ax.plot([points[1, 0], points[3, 0]], [points[1, 1], points[3, 1]], [points[1, 2], points[3, 2]])
ax.plot([points[1, 0], points[5, 0]], [points[1, 1], points[5, 1]], [points[1, 2], points[5, 2]])
ax.plot([points[2, 0], points[3, 0]], [points[2, 1], points[3, 1]], [points[2, 2], points[3, 2]])
ax.plot([points[2, 0], points[6, 0]], [points[2, 1], points[6, 1]], [points[2, 2], points[6, 2]])
ax.plot([points[3, 0], points[7, 0]], [points[3, 1], points[7, 1]], [points[3, 2], points[7, 2]])
ax.plot([points[4, 0], points[5, 0]], [points[4, 1], points[5, 1]], [points[4, 2], points[5, 2]])
ax.plot([points[4, 0], points[6, 0]], [points[4, 1], points[6, 1]], [points[4, 2], points[6, 2]])
ax.plot([points[5, 0], points[7, 0]], [points[5, 1], points[7, 1]], [points[5, 2], points[7, 2]])
ax.plot([points[6, 0], points[7, 0]], [points[6, 1], points[7, 1]], [points[6, 2], points[7, 2]])

plt.show()