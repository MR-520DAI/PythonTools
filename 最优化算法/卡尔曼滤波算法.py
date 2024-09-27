# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(1,100,100)
noise = np.random.randn(100)
z = z+noise   # 观测值+模拟噪声

X = np.array([[0.], 
              [0.]])
P = np.array([[1.,0.],
              [0.,1.]])
F = np.array([[1.,1.],
              [0.,1.]])
FT = F.transpose()
Q = np.array([[0.001, 0.],
              [0.,0.001]])
H = np.array([[1., 0.]])
HT = H.transpose()
R = 10.0

v_ = []
p_ = []

for i in range(1, 100, 1):
  X_ = np.matmul(F, X)
  P_ = np.matmul(np.matmul(F,P), FT) + Q
  K = np.matmul(P_, HT) * np.linalg.inv(np.matmul(np.matmul(H, P_), HT) + R)
  X = X_ + np.matmul(K, z[i]-np.matmul(H, X_))
  P = np.matmul(np.eye(2)-np.matmul(K, H), P_)
  v_.append(X[1][0])
  p_.append(X[0][0])

plt.plot(p_, v_)
plt.show()
