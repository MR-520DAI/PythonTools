import cv2
import numpy as np


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


if __name__ == '__main__':
    # rect1 = (661, 27, 679, 47)
    # # (top, left, bottom, right)
    # rect2 = (662, 27, 682, 47)
    # iou = compute_iou(rect1, rect2)
    # print(iou)
    pointcolor = (0, 255, 0)    #点的颜色
    reccolor = (0, 0, 255)      #框的颜色
    rec_w = 50                  #框的宽度
    rec_h = 50                  #框的高度

    img = cv2.imread("./data/paowuxian.jpg")
    #cv2.circle(img, (50, 50), 3, pointcolor, -1)

    points = [(50, 50),
              (50, 70),
              (51, 90),
              (52, 110),
              (50, 130),
              (49, 150),
              (52, 170),
              (50, 190),
              (48, 210)]
    lp = []
    lp.append(points[0])
    for i in range(1, len(points)):
        cv2.circle(img, lp[i-1], 3, pointcolor, -1)
        cv2.rectangle(img, (int(lp[i-1][0]-rec_w/2), int(lp[i-1][1]-rec_h/2)), (int(lp[i-1][0]+rec_w/2), int(lp[i-1][1]+rec_h/2)), reccolor)

        iou = compute_iou((lp[i-1][0]-rec_w/2, lp[i-1][1]-rec_h/2, lp[i-1][0]+rec_w/2, lp[i-1][1]+rec_h/2),
                          (points[i][0]-rec_w/2, points[i][1]-rec_h/2, points[i][0]+rec_w/2, points[i][1]+rec_h/2))
        if iou > 0.3:
            lp.append(points[i])
        if len(lp) > 2:
            l = np.array(lp, np.int32)
            cv2.polylines(img, [l], False, (255, 0, 0), 2)

        cv2.imshow("test", img)
        cv2.waitKey(0)
