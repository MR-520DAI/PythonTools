import os

root = "Y:\\ADAS\In\\Raw\\hiv00030_yuv_360_new\\"

file_name = os.listdir(root)

print(file_name[0][7:11])

for i in file_name:
    print(i)
    os.rename(root + i, root + i[7:11] + ".yuv")
