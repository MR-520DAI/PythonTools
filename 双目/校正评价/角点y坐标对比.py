import cv2

w = 9
h = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

imgl = cv2.imread("left01_MY.jpg")
grayl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
imgr = cv2.imread("right01_MY.jpg")
grayr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)

ret, cornersl = cv2.findChessboardCorners(grayl, (w, h))
if ret:
    cornersl = cv2.cornerSubPix(grayl, cornersl, (11, 11), (-1, -1), criteria)
    print(cornersl)

print("0000000000000000000000000000000000000000")

ret, cornersr = cv2.findChessboardCorners(grayr, (w, h))
if ret:
    cornersr = cv2.cornerSubPix(grayr, cornersr, (11, 11), (-1, -1), criteria)
    print(cornersr)

sum = 0
for i in range(54):
    imgl_y = cornersl[i][0][1]
    imgr_y = cornersr[i][0][1]
    sub_y = cornersl[i][0][1] - cornersr[i][0][1]
    sum += sub_y
    print("imgl_y:{0}, imgr_y:{1}, sub_y:{2}".format(imgl_y, imgr_y, sub_y))

print("mean:", (sum/54))