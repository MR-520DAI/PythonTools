import cv2
import numpy as np

def image_stitching(images):
    # 创建特征提取器
    orb = cv2.ORB_create()

    # 寻找特征点和计算特征描述符
    keypoints = []
    descriptors = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)

    # 特征点匹配
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = []
    for i in range(len(descriptors)-1):
        match = matcher.match(descriptors[i], descriptors[i+1])
        matches.append(match)

    # 图像配准
    aligned_images = []
    for i in range(len(images)-1):
        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i+1][m.trainIdx].pt for m in matches[i]]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print(M)
        aligned_img = cv2.warpPerspective(images[i], M, (images[i].shape[1], images[i].shape[0]))
        cv2.imwrite("data\\warpPerspective_" + str(i) + ".jpg", aligned_img)
        # aligned_images.append(aligned_img)

images = []
for i in range(3):
    img = cv2.imread("data\\" + str(i) + ".jpg")
    images.append(img)

image_stitching(images)