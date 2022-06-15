import cv2
import numpy as np

fx = 1407.30358747218
fy = 1400.52008918695
cx = 239.037469642279
cy = 179.175305653572
k1 = -0.445491944643704
k2 = 0.513647969014420
p1 = -0.00351212188197951
p2 = -0.00226950380736736
k3 = 0.0

P = np.matrix([[1.43415275e+03, 0.0, 3.51965359e+02],
               [0.0, 1.43415275e+03, 1.80466177e+02],
               [0.0, 0.0, 1.0]])
R = np.matrix([[0.99694379, -0.01004652, -0.07747348],
               [0.01018289, 0.99994722, 0.00136536],
               [0.07745568, -0.0021501, 0.99699348]])
PR = P*R
PR_I = PR.I
PR = np.array(PR)
PR_I = np.array(PR_I)
R = np.array(R)

# 不动点迭代
x_distortion = 268
y_distortion = 304

c1 = (x_distortion - cx) / fx
c2 = (y_distortion - cy) / fy

x_temp = c1
y_temp = c2

x_2 = x_temp * x_temp
y_2 = y_temp * y_temp
r_2 = x_2 + y_2
_2xy = 2 * x_temp * y_temp
kr = 1 + ((k3 * r_2 + k2) * r_2 + k1) * r_2

num = 0

while True:
    num += 1
    x_target = (c1 - p1 * _2xy - p2 * (r_2 + 2 * x_2)) / kr
    y_target = (c2 - p1 * (r_2 + 2 * y_2) - p2 * _2xy) / kr

    x_2 = x_target * x_target
    y_2 = x_target * y_target
    r_2 = x_2 + y_2
    _2xy = 2 * x_target * y_target
    kr = 1 + ((k3 * r_2 + k2) * r_2 + k1) * r_2

    print((c1 - p1 * _2xy - p2 * (r_2 + 2 * x_2)) / kr - x_target)
    print((c2 - p1 * (r_2 + 2 * y_2) - p2 * _2xy) / kr - y_target)

    if abs((c1 - p1 * _2xy - p2 * (r_2 + 2 * x_2)) / kr - x_target) < 0.00001 and abs((c2 - p1 * (r_2 + 2 * y_2) - p2 * _2xy) / kr - y_target) < 0.00001:
        break
    # else:
    #     x_temp = x_target
    #     y_temp = y_target

print("去除畸变:")
print(x_target * fx + cx)
print(y_target * fy + cy)

_x = y_target * R[0][1] + R[0][2]
_y = y_target * R[1][1] + R[1][2]
_w = y_target * R[2][1] + R[2][2]

_x += x_target * R[0][0]
_y += x_target * R[1][0]
_w += x_target * R[2][0]

x_target = _x / _w
y_target = _y / _w

x_target = x_target * 1.43415275e+03 + 3.51965359e+02
y_target = y_target * 1.43415275e+03 + 1.80466177e+02

print("映射后")
print(x_target)
print(y_target)

print(num)