import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

img1_name = "data\\img\\5.jpg"
img2_name = "data\\img\\6.jpg"

img1 = load_torch_image(img1_name)
img2 = load_torch_image(img2_name)

matcher = KF.LoFTR(pretrained='indoor')

input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR 只在灰度图上作用
              "image1": K.color.rgb_to_grayscale(img2)}
with torch.inference_mode():
    correspondences = matcher(input_dict)
for k,v in correspondences.items():
    print (k)
# 现在让我们用现代RANSAC清理对应关系，并估计两幅图像之间的基本矩阵
mkpts0 = correspondences['keypoints0'].cpu().numpy()
mkpts1 = correspondences['keypoints1'].cpu().numpy()
# print(mkpts0)
# print(mkpts1)

H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
print(H)
img1_color = cv2.imread("data\\img\\5.jpg")
img1Reg = cv2.warpPerspective(img1_color, H, (img1_color.shape[1], img1_color.shape[0]))
cv2.imwrite('data\\img\\rst.jpg', img1Reg)