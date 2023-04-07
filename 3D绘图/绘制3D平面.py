import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax1 = Axes3D(fig)

plane_set = []
plane1_map = [0.0342072, 0.0106903, -0.999358, 1.49954]
plane2_map = [-0.336846, -0.940937, 0.0342504, 1.80646]
plane3_map = [-0.943887, 0.324757, -0.0600812, 2.09947]
plane4_map = [-0.316135, -0.948687, -0.00717792, -3.12093]
plane5_map = [0.0569872, 0.0216719, -0.99814, -0.429842]
plane_set.append(plane1_map)
plane_set.append(plane2_map)
plane_set.append(plane3_map)
plane_set.append(plane4_map)
plane_set.append(plane5_map)

for i in plane_set:
    a = i[0]
    b = i[1]
    c = i[2]
    d = i[3]
    if abs(a/d) > abs(b/d) and abs(a/d) > abs(c/d):
        Y = np.arange(-5, 5, 0.2)
        Z = np.arange(-5, 5, 0.2)
        Y, Z = np.meshgrid(Y, Z)
        X = -(b*Y+c*Z+d)/a
        ax1.plot_surface(X, Y, Z, rstride=1, cstride=1)
    elif abs(b/d) > abs(a/d) and abs(b/d) > abs(c/d):
        X = np.arange(-5, 5, 0.2)
        Z = np.arange(-5, 5, 0.2)
        X, Z = np.meshgrid(X, Z)
        Y = -(a*X+c*Z+d)/b
        ax1.plot_surface(X, Y, Z, rstride=1, cstride=1)
    else:
        X = np.arange(-5, 5, 0.2)
        Y = np.arange(-5, 5, 0.2)
        X, Y = np.meshgrid(X, Y)
        Z = -(a*X+b*Y+d)/c
        ax1.plot_surface(X, Y, Z, rstride=1, cstride=1)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('plane vis')
plt.show()
