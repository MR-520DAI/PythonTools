import cv2
import time
import torch
import numpy as np
import kornia as K
import kornia.feature as KF
from kornia.feature.adalam import AdalamFilter

def ORB(img1path, img2path):
    img1 = cv2.imread(img1path,0) # 查询图片  
    img2 = cv2.imread(img2path,0) # 训练图片
    orb = cv2.ORB_create()  

    # 寻找关键点  
    kp1, des1 = orb.detectAndCompute(img1, None)  
    kp2, des2 = orb.detectAndCompute(img2, None)  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
    matches = bf.match(des1,des2)

    output_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    matches = sorted(matches, key = lambda x:x.distance)

    points1 = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)  
    points2 = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    return H

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
    kp0 = []
    kp1 = []
    for i in range(len(confidence)):
        if confidence[i] > 0.99:
            kp0.append([mkpts0[i][0], mkpts0[i][1]])
            kp1.append([mkpts1[i][0], mkpts1[i][1]])
    H, mask = cv2.findHomography(np.array(kp0), np.array(kp1), cv2.RANSAC)
    img1 = cv2.imread(img1path)   # 特征点可视化
    img2 = cv2.imread(img2path)
    kp0 = []
    num = 0
    for i in mkpts0:
        if confidence[num] > 0.99:
            kp0.append(cv2.KeyPoint(i[0], i[1], 1))
            num += 1
        else:
            num += 1
            continue
        
    kp1 = []
    num = 0
    for i in mkpts1:
        if confidence[num] > 0.99:
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
    return H

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2

def DISKFeatureMatch(img1path, img2path):
    device = K.utils.get_cuda_or_mps_device_if_available()
    img1 = K.io.load_image(img1path, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    img2 = K.io.load_image(img2path, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    num_features = 2048
    disk = KF.DISK.from_pretrained("depth").to(device)

    # hw1 = torch.tensor(img1.shape[2:], device=device)
    # hw2 = torch.tensor(img2.shape[2:], device=device)
    with torch.inference_mode():
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = disk(inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors
        dists, idxs = KF.match_smnn(descs1, descs2, 0.98)
        mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)
    # kps1_numpy = kps1.detach().numpy()
    # descs1_numpy = descs1.detach().numpy()
    # kps2_numpy = kps2.detach().numpy()
    # descs2_numpy = descs2.detach().numpy()
    if mkpts1.size()[0] < 200:
        H = np.array([[1.0, 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        return H
    H, mask = cv2.findHomography(mkpts1.detach().numpy(), mkpts2.detach().numpy(), cv2.RANSAC)
    return H

def DISKFeature(img1path, disk):
    device = K.utils.get_cuda_or_mps_device_if_available()
    img1 = K.io.load_image(img1path, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    with torch.inference_mode():
        features1 = disk(img1)
        kps_x = features1[0].x
        kps_y = features1[0].y
        des = features1[0].descriptors
    return kps_x.detach().numpy(), kps_y.detach().numpy(), des.detach().numpy()

if __name__ == "__main__":
    # 相机参数
    fx = 406.1814  # 焦距x（单位像素）
    fy = 405.7139  # 焦距y（单位像素）
    cx = 311.5565  # 光心x
    cy = 250.0853  # 光心y

    root = "data\\img\\"
    matcher = KF.LoFTR(pretrained='indoor')     # 检测器
    start_time = time.time()
    # for i in range(1, 38, 1):
    #     img1_path = root + str(i) + ".jpg"
    #     img2_path = root + str(i+1) + ".jpg"
    #     H = LoFTR(img1_path, img2_path, matcher)
    #     img1 = cv2.imread(img1_path)
    #     img1Reg = cv2.warpPerspective(img1, H, (640, 480))
    #     cv2.imwrite(root + str(i+1) + "_reg.jpg", img1Reg)
    #     print(root + str(i+1) + "_reg.jpg")
    img1_path = "E:\\data\\MYData\\CamLidardata\\0.jpg"
    img2_path = "E:\\data\\MYData\\CamLidardata\\18.jpg"


    H = LoFTR(img1_path, img2_path, matcher)
    H = DISKFeatureMatch(img1_path, img2_path)
    print("H:\n", H)
    K = np.array([[fx, cx, 0.], [0., fy, cy], [0.,0.,1.]])
    num, R, t, Ns = cv2.decomposeHomographyMat(H, K)
    print("R:\n", R)
    print("t:\n", t)
    img1 = cv2.imread(img1_path)
    img1Reg = cv2.warpPerspective(img1, H, (640, 480))
    cv2.imwrite("E:\\data\\MYData\\CamLidardata\\reg.jpg", img1Reg)

    # disk = KF.DISK.from_pretrained("depth").to("cpu")
    # for i in range(1):
    #     img1_path = "E:\\data\\MYData\\CamLidardata\\" + str(i) + ".jpg"
    #     kpx, kpy, des = DISKFeature(img1_path, disk)
    #     txt_path = img1_path + ".txt"
    #     with open(txt_path, "w") as f:
    #         head = str(kpx.shape[0]) + " 128\n"
    #         f.write(head)
    #         for x, y in zip(kpx, kpy):
    #             f.write(str(x) + " " + str(y) + "\n")
    
    end_time = time.time()
    print("程序执行时间为：{}秒".format(end_time - start_time))
