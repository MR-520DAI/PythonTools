k1 = -0.286739
k2 = -0.0337054
p1 = -0.00005977
p2 = -0.001538837
k3 = 0.10722879

# 不动点迭代法

x_distortion = 1331
y_distortion = 882

c1 = (x_distortion - 1437.86) / 1892.6
c2 = (y_distortion - 822.36) / 1892.6

x_temp = (x_distortion - 1437.86) / 1892.6
y_temp = (y_distortion - 822.36) / 1892.6

while True:
    r = x_temp**2 + y_temp**2
    x = (c1 - 2*p1*x_temp*y_temp - p2*(3*x_temp**2+y_temp**2)) / (1+k1*r+k2*r**2+k3*r**3)
    y = (c2 - 2*p2*x_temp*y_temp - p1*(3*y_temp**2+x_temp**2)) / (1+k1*r+k2*r**2+k3*r**3)
    r = x**2 + y**2
    print(r)
    if (c1 - 2*p1*x*y - p2*(3*x**2+y**2)) / (1+k1*r+k2*r**2+k3*r**3) - x < 0.0001 and (c2 - 2*p2*x*y - p1*(3*y**2+x**2)) / (1+k1*r+k2*r**2+k3*r**3) - y < 0.0001:
        break
    else:
        x_temp = x
        y_temp = y

print(x*1892.6 + 1437.86)
print(y*1892.6 + 822.36)
