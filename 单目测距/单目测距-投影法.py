import math
import numpy as np

# 俯仰角向下
# theta1 = np.arctan(1400 / 930)
# theta2 = np.arctan(822 / 1892.6)
# theta3 = np.arctan((822.3-1001) / 1892.6)
#
# # print("theta1+theta2 = ", theta1 * 180 / 3.141592 + theta2 * 180 / 3.141592)
# print("theta1+theta2 = ", (np.pi / 2 - np.arctan((822-878)/1892.6)) * 180 / 3.141592)
# # print(theta2 * 180 / 3.141592)
# print("theta3 = ", theta3 * 180 / 3.141592)
#
# w = np.arctan((-(1382-1440)/1892.6) * np.cos(np.arctan((822-878)/1892.6)))
#
# d = 930 * np.tan(np.pi / 2 - np.arctan((800-878)/1892.6) + theta3) * np.cos(w)
# print(d)

# 俯仰角向上
# theta1_2 = np.pi / 2 - np.arctan((800-890)/1892.6)
# theta3 = np.arctan((1600 - 1034 - 822.3) / 1892.6)
# w = np.arctan((-(1336-1440)/1892.6) * np.cos(np.arctan((800-890)/1892.6)))
# d = 933 * np.tan(theta1_2 + theta3) * np.cos(w)
# print(d)

theta1_2 = np.pi / 2 - np.arctan((822-878)/1892.6)
print(theta1_2 * 180 / np.pi)
theta3 = np.arctan((822 - 965) / 1892.6)
print(theta3 * 180 / np.pi)
w = np.arctan(((1437.8 - 1298)/1892.6) * np.cos(np.arctan((822-878)/1892.6)))
d = 933 * np.tan(theta1_2 + theta3) * np.cos(w)
print(d)
