import os

for i in range(1, 5940):
    with open("1.txt", "a+") as f:
        f.write("hiv00030" + "/hiv00030" + str(i).zfill(3) + ".jpg\n")
        print(i)
