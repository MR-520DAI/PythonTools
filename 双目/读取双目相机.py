import cv2
import numpy as np

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
i = 0

while cap.isOpened():
    ret, frame = cap.read()
    imgl = frame[:, 0:1280, :]
    imgr = frame[:, 1280:2560, :]
    if ret:
        cv2.imshow("left", imgl)
        cv2.imshow("right", imgr)
    key = cv2.waitKey(delay=2)
    if key == ord('s'):
        cv2.imwrite('./test' + str(i) + '.jpg', frame)
        i += 1
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()