import open3d as o3d
import numpy as np

A, B, C, D = -0.682113,-0.709696,0.176217,1.1139
plane1 = [[ 2.41453, -0.962303,  0.628738],
          [0.348894, -3.27973,  1.13946],
          [1.44527, 0.297442, 0.854917],
          [-1.10141, -1.88949,  1.51512],
          [2.10392, -0.830681,  -1.35626],
          [2.61857, -0.800908,  0.559341],
          [2.19373, -0.800235,  -1.01259],
          [0.405127,  1.29233, 0.531796],
          [-0.0838124,   -3.27429,   -0.65959],
          [0.219268,  -3.3653,  1.11948],
          [-0.353895,  -3.30631,  -1.12909],
          [2.60055, -0.941136,  0.460782],
          [-0.114705,  -3.27699, -0.615282],
          [ -1.2969, -1.78564,  1.47409],
          [-0.3814, -3.28124,  -1.1181],
          [0.288336, -3.30691,  1.17321]]

# 创建立方体的顶点坐标
vertices = np.array(plane1)

# 创建立方体的面的索引
faces = np.array([[0,1,2],
                 [1,2,3],
                 [4,5,6],
                 [5,6,7],
                 [8,9,10],
                 [9,10,11],
                 [12,13,14],
                 [13,14,15]])

# 创建 Open3D 的几何图形对象
cube = o3d.geometry.TriangleMesh()
cube.vertices = o3d.utility.Vector3dVector(vertices)
cube.triangles = o3d.utility.Vector3iVector(faces)

# 显示立方体
o3d.visualization.draw_geometries([cube])
o3d.io.write_triangle_mesh("cube.ply", cube)