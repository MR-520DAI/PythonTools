import numpy as np

h = 580
d = 4533
fx = 1892.602
fy = 1892.602
cx = 1437.864
cy = 882.368

x = 1477
y = 1135

thetaC = np.arctan((y - cy) / fy)
print(thetaC * 180 / np.pi)
theta = np.arctan(d/h) - thetaC

print(theta)
