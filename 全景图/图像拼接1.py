import cv2
import numpy as np

def orb_stitch(image1, image2):
    # 创建ORB对象
    orb = cv2.ORB_create()

    # 在两张图像上检测特征点和计算描述符
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # 创建BFMatcher对象并进行特征点匹配
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    # 筛选最佳匹配点
    # matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点的位置
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算投影矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
    print(M)

    # 对image1进行透视变换
    h, w = image1.shape[:2]
    result = cv2.warpPerspective(image1, M, (w + image2.shape[1], h))

    # 将image2拼接到结果图像中
    result[:image2.shape[0], w:w + image2.shape[1]] = image2

    return result

# 加载图像
image1 = cv2.imread('data\\1.jpg')
image2 = cv2.imread('data\\2.jpg')

# 图像拼接
result = orb_stitch(image1, image2)
cv2.imwrite("result.jpg", result)

# 显示结果
# cv2.imshow('Stitched Image', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()