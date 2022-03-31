# -*- coding: utf-8 -*-

import cv2
import json
import numpy as np

img = cv2.imread("data/000a706ea929755e6bf6583ff2ce4a81.jpg")

area = []

with open("data/000a706ea929755e6bf6583ff2ce4a81.json", "r", encoding="utf8") as f:
    json_data = json.load(f)
    for ipolygon in json_data["shapes"]:
        area.append(np.array(ipolygon["points"], dtype=np.int32))
print(area)

cv2.fillPoly(img, area, (255,255,255))
cv2.imwrite("2.jpg", img)