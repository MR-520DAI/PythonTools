import os
import cv2

img_name = os.listdir("data\\scene-duomeiti")
img_name.sort(reverse=True)
print(img_name)

num = 0
for i in img_name:
    img = cv2.imread("data\\scene-duomeiti\\" + i)
    cv2.imwrite("data\\tmp\\" + str(num) + ".jpg", img)
    num += 1
    print(num)