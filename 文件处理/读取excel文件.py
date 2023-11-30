import pandas as pd

df = pd.read_excel("C:\\Users\\Administrator\\Desktop\\tmpdata\\标定数据\\工作簿1.xlsx")

headlines = [["L1", "U1", "V1"],
             ["L2", "U2", "V2"],
             ["L3", "U3", "V3"],
             ["L4", "U4", "V4"],
             ["L5", "U5", "V5"],
             ["L6", "U6", "V6"],
             ["L7", "U7", "V7"],
             ["L8", "U8", "V8"],
             ["L9", "U9", "V9"]]

loc = 0

for line in range(22, 30, 1):
    with open(str(line)+".txt", "w") as f:
        for head in headlines:
            if int(df[head[1]][line]) == 0:
                continue
            data = str(df[head[0]][line]/1000) + "," + str(df[head[1]][line]) + "," + str(df[head[2]][line]) + "\n"
            # print(data)
            f.write(data)