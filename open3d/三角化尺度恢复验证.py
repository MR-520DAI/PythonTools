import cv2
import numpy as np

def pixel2cam(K, imgpoint, depth):
    campoint = np.matmul(np.linalg.inv(K), imgpoint)
    campoint *= depth
    return campoint

K = np.array([[518.96, 0., 320.55],
              [0., 518.87, 237.88],
              [0.,0.,1.]])
color_img = cv2.imread("E:\\c_project\\tools\\COLMAP-3.8\\project\\images\\23.jpg")
depth_img = cv2.imread("E:\\c_project\\tools\\COLMAP-3.8\\project\\depth\\23.png", cv2.IMREAD_UNCHANGED)

imgpoint_list = [[222,181], [217,160],
                 [498,162], [526,232]]

for i in range(len(imgpoint_list)):
    depth = depth_img[int(imgpoint_list[i][1])][int(imgpoint_list[i][0])] / 1000
    imgpoint = np.array([[imgpoint_list[i][0]], 
                         [imgpoint_list[i][1]], 
                         [1.0]])
    point = pixel2cam(K, imgpoint, depth)
    print(point)
