import cv2
import time
from matplotlib import pyplot as plt

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("相机未正常工作，请检查串口连接")
else:
    print("相机正常工作")

num = 0
while True:
    ret, frame = cap.read()
    num += 1
    time.sleep(1)
    if num == 10:
        break

plt.imshow(frame)
plt.show()