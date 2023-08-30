import numpy as np

# 平面方程定义
def compute_plane_normal(A, B, C, D):
    # 计算法向量
    normal = np.array([A, B, C])
    return normal

# 计算夹角
def compute_angle_between_planes(plane1, plane2):
    # 法向量归一化
    plane1_normalized = plane1 / np.linalg.norm(plane1)
    plane2_normalized = plane2 / np.linalg.norm(plane2)
    
    # 计算夹角的余弦值
    cos_theta = np.dot(plane1_normalized, plane2_normalized)
    
    # 计算夹角的度数
    theta = np.arccos(cos_theta)
    
    # 将弧度转换为角度
    angle_degrees = np.degrees(theta)
    
    return angle_degrees

# 示例平面方程
A1, B1, C1, D1 = -0.333008,0.223288,-0.916105,1.52755
A2, B2, C2, D2 = 0.00146407,-0.970998,-0.239082,0.813396

# 计算平面1的法向量
plane1_normal = compute_plane_normal(A1, B1, C1, D1)

# 计算平面2的法向量
plane2_normal = compute_plane_normal(A2, B2, C2, D2)

# 计算夹角
angle = compute_angle_between_planes(plane1_normal, plane2_normal)

print("夹角：", angle)
