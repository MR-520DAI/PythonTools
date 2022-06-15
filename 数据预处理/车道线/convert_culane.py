import os
import cv2
import numpy as np

root = "Y:/data/CULane"

def draw(LabelPath, Idx):
    LabelImg = np.zeros((590, 1640), dtype=np.int8)
    if Idx == [0, 0, 0, 0]:
        return LabelImg
    else:
        id = 0
        color = 0
        with open(LabelPath, 'r') as f:
            Pst = f.readlines()
            for i in range(len(Pst)):
                Lane = Pst[i].strip("\n").split(" ")
                # print(Lane)
                l = []
                for j in range(0, len(Lane) - 1, 2):
                    px = float(Lane[j])
                    py = float(Lane[j+1])
                    if py < 0.:
                        continue
                    else:
                        l.append([int(px), int(py)])
                l = np.array(l)
                for k in range(id, len(Idx)):
                    if Idx[k] != 0:
                        color = Idx[k]
                        break
                    else:
                        continue

                LabelImg = cv2.polylines(LabelImg, [l], False, color, 16)
                id = color
        return LabelImg


with open(root + "/list/train_gt.txt", "r") as f:
    trainlist = f.readlines()
    for i in range(len(trainlist)):
        print(i)
        Idx = [0, 0, 0, 0]
        ImgPath = root + trainlist[i].split(" ")[0]
        LableName = trainlist[i].split(" ")[0].split("/")[-1].split(".")[0] + ".lines.txt"
        LabelPath = root + trainlist[i].split(" ")[0][0:-9] + LableName
        # print(ImgPath)
        # print(LabelPath)

        if trainlist[i].split(" ")[2] == '1':
            Idx[0] = 1
        else:
            Idx[0] = 0

        if trainlist[i].split(" ")[3] == '1':
            Idx[1] = 2
        else:
            Idx[1] = 0

        if trainlist[i].split(" ")[4] == '1':
            Idx[2] = 3
        else:
            Idx[2] = 0

        if trainlist[i].split(" ")[5].strip('\n') == '1':
            Idx[3] = 4
        else:
            Idx[3] = 0

        LabelImg = draw(LabelPath, Idx)
        cv2.imwrite(root + "/laneseg_label_w16" + trainlist[i].split(" ")[0][0:-4] + ".png", LabelImg)

        print(root + "/laneseg_label_w16" + trainlist[i].split(" ")[0][0:-4] + ".png")
        # print("*********************")

