# coding=utf8
import cv2
import numpy as np

def EqualCylindricalProjection(NumImg, Img, f=-1):
    """生成等距柱形投影图

    Args:
        NumImg (int): 图像序列的总数
        Img (CV Mat): 源图像数组
        f (float): 相机焦距
    Returns:
        Numpy Mat: 等距柱形投影图数组
    """
    H, W, C = Img.shape
    result = np.zeros((H,W,C), dtype=np.uint8)

    if f == -1:     # 焦距估计
        alpha = 2 * np.pi / NumImg
        f = W / (2*np.tan(alpha/2))
    else:
        pass
    w = 2*np.arctan(W / 2 / f)
    for i in range(H):
        print(i)
        for j in range(W):
            d = np.sqrt(f*f + (W/2 - j)*(W/2 - j))
            x = f*np.tan(j/f-w/2) + W/2
            y = d*(i-H/2)/f + H/2
            if x > 0 and x < (W-1) and y > 0 and y < (H-1):
                u = x - int(x)
                v = y - int(y)
                s1 = Img[int(y)][int(x)]
                s2 = Img[int(y)][int(x)+1]
                s3 = Img[int(y)+1][int(x)]
                s4 = Img[int(y)+1][int(x)+1]
                s0 = (1-u)*(1-v)*s1+(1-u)*v*s2+u*(1-v)*s3+u*v*s4
                result[i][j] = s0
    
    return result

if __name__ == "__main__":
    Img = cv2.imread("E:\\py_project\\0.jpg")
    rst = EqualCylindricalProjection(38, Img)
    cv2.imwrite("1.jpg", rst)
