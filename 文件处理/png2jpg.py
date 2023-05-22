import os
import cv2

root = "E:/data/TUM/rgbd_dataset_freiburg3_long_office_household/"

imgName = os.listdir(root + "color")

for i in imgName:
    img = cv2.imread(root + "color/" + i)
    print(root + "color1/" + i.split(".")[0] + ".jpg")
    cv2.imwrite(root + "color1/" + i.split(".")[0] + ".jpg", img)