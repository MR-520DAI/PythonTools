# -*- coding: utf-8 -*-

import os
import cv2

img_w = 1920
img_h = 1080

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vout = cv2.VideoWriter('D:\\c_project\\dms\\dms\\IA_TEST\\video\\old.avi', fourcc, 30.0, (img_w, img_h))

Id = 28

while(Id < 5938):
    img = cv2.imread("D:\\c_project\\dms\\dms\\IA_TEST\\pic\\" + str(Id) + ".jpg")
    vout.write(img)
    Id += 1
    print(Id)
