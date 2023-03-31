import pandas as pd

df = pd.read_excel("1.xlsx")

for index, row in df.iterrows():
    if index % 9 == 0:
        print("*********************")
    print("{", row[9], ",",row[10], ",",row[11], ",",row[12], ",",row[13], ",",row[14], ",","},")