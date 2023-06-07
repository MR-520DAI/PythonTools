import cv2

nameList = [0.541, 1.069, 1.583, 2.036, 2.534, 2.997, 3.491, 4.012, 4.548, 5.097]

# 手动选择平面区域像素坐标：左上角（x,y）、右下角(x,y)
ROIList = [[232, 171, 424, 315],
           [181, 129, 466, 332],
           [198, 136, 443, 332],
           [180, 128, 421, 306],
           [170, 111, 411, 290],
           [189, 108, 396, 271],
           [203, 106, 390, 255],
           [201, 111, 389, 249],
           [186, 93, 381, 248],
           [212, 108, 368, 251]]

for i in range(len(nameList)):
    depthImg = cv2.imread("./data/" + str(nameList[i]) + ".png", cv2.IMREAD_UNCHANGED)
    num = 0
    depthRGBD = 0
    for y in range(ROIList[i][1], ROIList[i][3], 1):
        for x in range(ROIList[i][0], ROIList[i][2], 1):
            num += 1
            depth = depthImg[y][x] / 1250
            depthRGBD += depth
    depthRGBD /= num
    rate = abs(depthRGBD - nameList[i]) / nameList[i]
    print("真值距离: ", nameList[i], " rgbd距离: ", depthRGBD, " 误差比例: ", rate)
