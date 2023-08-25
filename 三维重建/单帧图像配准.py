import cv2
import torch
import numpy as np
import kornia as K
import kornia.feature as KF

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

def LoFTR(img1path, img2path, matcher):
    img1 = load_torch_image(img1path)
    img2 = load_torch_image(img2path)
    # matcher = KF.LoFTR(pretrained='indoor')
    input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR 只在灰度图上作用
              "image1": K.color.rgb_to_grayscale(img2)}
    with torch.inference_mode():
        correspondences = matcher(input_dict)
    # for k,v in correspondences.items():
    #     print (k)
    
    # 现在让我们用现代RANSAC清理对应关系，并估计两幅图像之间的基本矩阵
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()
    # kp0 = []
    # kp1 = []
    # for i in range(len(confidence)):
    #     if confidence[i] > 0.9:
    #         kp0.append([mkpts0[i][0], mkpts0[i][1]])
    #         kp1.append([mkpts1[i][0], mkpts1[i][1]])
    # H, mask = cv2.findHomography(np.array(kp0), np.array(kp1), cv2.RANSAC)
    img1 = cv2.imread(img1path)   # 特征点可视化
    img2 = cv2.imread(img2path)
    kp0 = []
    num = 0
    for i in mkpts0:
        if confidence[num] > 0.95:
            kp0.append(cv2.KeyPoint(i[0], i[1], 1))
            num += 1
        else:
            num += 1
            continue
        
    kp1 = []
    num = 0
    for i in mkpts1:
        if confidence[num] > 0.95:
            kp1.append(cv2.KeyPoint(i[0], i[1], 1))
            num += 1
        else:
            num += 1
            continue
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
    output_img = cv2.drawMatches(img1, kp0, img2, kp1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return H
if __name__ == "__main__":
    root = "data\\img\\"
    matcher = KF.LoFTR(pretrained='indoor')     # 检测器

    img1_path = root + "6.jpg"
    img2_path = root + "9.jpg"
    LoFTR(img1_path, img2_path, matcher)
