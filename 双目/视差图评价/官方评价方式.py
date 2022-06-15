import cv2
import math

ABS_THRESH = 3.
REL_THRESH = 0.05

def disparityErrorsOutlier(D_gt, D_orig):

    errors = []
    num_errors_all = 0
    num_pixels_all = 0
    num_errors_all_result = 0
    num_pixels_all_result = 0

    for i in range(D_gt.shape[0]):
        for j in range(D_gt.shape[1]):
            if D_gt[i][j] != 0:
                d_err = math.fabs(float(D_gt[i][j]) - float(D_orig[i][j])) > ABS_THRESH and \
                        math.fabs(float(D_gt[i][j]) - float(D_orig[i][j])) / math.fabs(float(D_gt[i][j])) > REL_THRESH
                if d_err:
                    num_errors_all += 1
                num_pixels_all += 1
                if D_orig[i][j] != 0:
                    if d_err:
                        num_errors_all_result += 1
                    num_pixels_all_result += 1

    errors.append(num_errors_all)   # 总体出错个数
    errors.append(num_pixels_all)   # 真值个数
    errors.append(num_errors_all_result)    # 有效（真值和预测同时不为0）出错个数
    errors.append(num_pixels_all_result)    # 有效个数
    errors.append(num_pixels_all_result / max(num_pixels_all, 1))   # 有效率
    errors.append(num_errors_all_result / max(num_pixels_all_result, 1))    # 错误率

    return errors

if __name__ == '__main__':
    Img_gt = cv2.imread("000121_10_disp.jpg", 0)
    Img_pre = cv2.imread("000121_10_disp_true.jpg", 0)

    errors = disparityErrorsOutlier(Img_gt, Img_pre)
    print("真值总数:", errors[1])
    print("有效总数:", errors[3])
    print("有效出错总数:", errors[2])
    print("有效率:", errors[4])
    print("有效出错率:", errors[5])