import os
import cv2
import numpy as np

def undistort(img_path,K,D,DIM,scale=1.0,imshow=True):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0]!=DIM[0]:
        img = cv2.resize(img,DIM,interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:#change fov
        Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if imshow:
        cv2.imshow("undistorted", undistorted_img)
        cv2.waitKey(0)
    return undistorted_img

if __name__ == '__main__':
    '''
    DIM, K, D = get_K_and_D((6,9), '')
    '''
    DIM=(256, 192)
    K=np.array([[121.35078337502344, 0.0, 134.76764230651165], [0.0, 121.23663274751947, 97.95035796932528], [0.0, 0.0, 1.0]])
    D=np.array([[0.09515957040074036], [-0.3171749462967648], [0.409069961617535], [-0.17746697055901417]])

    img = undistort('E:\\data\\lingguang\\calib\\0.png',K,D,DIM)
    cv2.imwrite('pig_checkerboard.jpg', img)
