import os
import cv2

root = "/home/dzy/data/MYdata/20221201/mav0/cam1/data/"

name_list = os.listdir(root)

for i in name_list:
    img = cv2.imread(root + i, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(root + i, gray)
    print(i)