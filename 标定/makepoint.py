import numpy as np
import pandas as pd

calib = [[-6.16584, -25.6793, 27.4531, -1.12252, -0.221863, 0.366025],
         [-8.69913, 37.5231, 32.5295, -1.07933, 0.265711, 0.361823],
         [-12.7283, -26.8757, -33.204, -1.11884, -0.221979, -0.359693],
         [-4.08787, 36.8015, -28.5265, -1.13437, 0.27732, -0.30958]]

df_ID = [1, 3, 7, 9]
df = pd.read_excel("C:\\Users\\Administrator\\Desktop\\tmpdata\\data-1.xlsx")
col_list = list(df.columns)
for i in range(8):
    for j in range(4):
        L = df[col_list[df_ID[j]]][i] / 1000
        X = -(calib[j][0] / 1000 + L * np.sin(calib[j][3]))
        Y = -(calib[j][1] / 1000 + L * np.sin(calib[j][4]))
        Z = -(calib[j][2] / 1000 + L * np.sin(calib[j][5]))
        print("[", X, ",", Y, ",", Z, "],")