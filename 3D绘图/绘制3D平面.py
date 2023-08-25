import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax1 = Axes3D(fig)

plane_set = []
plane1_map = [-0.0538746 ,0.012239 ,0.998473 ,-0.922939]
plane2_map = [0.018447 ,0.999676 ,-0.017548 ,1.63135]
plane3_map = [0.0199784 ,0.998334 ,-0.0541366 ,-1.1185]
plane4_map = [-0.999218 ,-0.00701429 ,-0.038925 ,1.1937]
plane5_map = [-0.998383 ,0.029792 ,-0.0484061 ,-1.5029]
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
