# -*- coding: utf-8 -*-

import os
import cv2

cap = cv2.VideoCapture("Y:\\data\\测距数据\\东\\hiv00030.mp4")
res, frame = cap.read()
while not res:
    res, frame = cap.read()
Id = 0
while res:
    #frame = cv2.resize(frame, (640, 360))
    cv2.imwrite("D:\\c_project\\dms\\dms\\IA_TEST\\In\\Jpg\\" + str(Id) + ".jpg", frame)
    Id += 1
    print(Id)
    res, frame = cap.read()
