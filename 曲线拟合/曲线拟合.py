import numpy as np

x = np.array([255,235,225,210,200,180,170,155,135,120,110,100,80,65,50], dtype=np.float32)
y = np.array([210,220,230,240,250,260,270,280,290,300,310,320,330,340,350], dtype=np.float32)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

print(p)