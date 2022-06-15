import cv2
import numpy as np
# M1
cameraMatrixL = [ 3.0014669374266246e+03, 0., 6.5437802195400036e+02, 0.,
       2.9946408852871764e+03, 3.3576534950910417e+02, 0., 0., 1. ]
cameraMatrixL = np.array(cameraMatrixL).reshape((3, 3))
# D1
cameraDistcoeffL = [ -6.1278626313247819e-01, 4.7073446829722245e+00,
       -2.0764724306749039e-03, -1.0167474909255332e-02, 0. ]
cameraDistcoeffL = np.array(cameraDistcoeffL).reshape((1, 5))

# M2
cameraMatrixR = [ 3.0001030914247945e+03, 0., 6.7264192673391346e+02, 0.,
       2.9880520653930571e+03, 3.3701117748860588e+02, 0., 0., 1. ]
cameraMatrixR = np.array(cameraMatrixR).reshape((3, 3))

#D2
cameraDistcoeffR = [ -5.1592540463450631e-01, 3.0994288511253759e-01,
       7.9022980317248472e-04, -7.6279446480335810e-03, 0. ]
cameraDistcoeffR = np.array(cameraDistcoeffR).reshape((1, 5))


R = [ 9.9983262426823694e-01, -1.7012691672521008e-02,
       -6.7299161173364191e-03, 1.7065997761645454e-02,
       9.9982280449895200e-01, 7.9442636065254028e-03,
       6.5935702991749834e-03, -8.0577866629854382e-03,
       9.9994579698341834e-01]
R = np.array(R).reshape((3, 3))

T = [ -1.7569717400990316e+02, -1.6795678537461614e+00,
       -1.0509855481236771e+01 ]
T = np.array(T)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrixL, cameraDistcoeffL, cameraMatrixR, cameraDistcoeffR,
                                                      (1080, 720), R, T, alpha=0)

map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrixL, cameraDistcoeffL, R1, P1, (1080, 720), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrixR, cameraDistcoeffR, R2, P2, (1080, 720), cv2.CV_32FC1)

for i in range(0, 2):
    imgl = cv2.imread("./stereo_calib/sample/left/" + str(i) + ".jpg")
    #imgl = cv2.resize(imgl, (1080, 720))
    imgr = cv2.imread("./stereo_calib/sample/right/" + str(i) + ".jpg")
    #imgr = cv2.resize(imgr, (1080, 720))
    rectifyl = cv2.remap(imgl, map1x, map1y, cv2.INTER_AREA)
    rectifyr = cv2.remap(imgr, map2x, map2y, cv2.INTER_AREA)
    cv2.imwrite("./stereo_calib/sample/left_rectify/" + str(i) + ".jpg", rectifyl)
    cv2.imwrite("./stereo_calib/sample/right_rectify/" + str(i) + ".jpg", rectifyr)

