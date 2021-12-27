import cv2
from matplotlib import pyplot as plt

# Canny算子进行边缘提取
img = cv2.imread("data\\RGB-face.jpg", 1)
img = cv2.GaussianBlur(img, (3, 3), 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img, 80, 160)

cv2.imwrite("data\\RGB-face-canny.jpg", canny)


'''
Sobel算子进行边缘提取
img = cv2.imread("ir-face.jpg", 0)
x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=5)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

cv2.imwrite("ir-face-sobel.jpg", dst)
'''

'''
Laplacian算子进行边缘提取
img = cv2.imread("ir-face.jpg", 0)
lap = cv2.Laplacian(img, cv2.CV_16S, ksize=5)
dst = cv2.convertScaleAbs(lap)

cv2.imwrite("ir-face-Lap.jpg", dst)
'''

#plt.imshow(dst)
#plt.show()