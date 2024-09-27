import cv2

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
corners_vertical = 6    # 纵向角点个数;
corners_horizontal = 7  # 横向角点个数;
pattern_size = (corners_vertical, corners_horizontal)

def calib(img_path, num):
  for i in range(1, num+1, 1):
      img_file = img_path + str(i) + ".png"
      src_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
      process_img = cv2.equalizeHist(src_img)
      median_img = cv2.medianBlur(process_img, 5)
      ret, corners = cv2.findChessboardCornersSB(process_img, pattern_size, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
      cv2.imwrite("E:\\data\\lingguang\\"+str(i)+".png", process_img)
      if ret:
        # 精细查找角点
        corners2 = cv2.cornerSubPix(process_img, corners, (5, 5), (-1, -1), criteria)
        # 显示角点
        cv2.drawChessboardCorners(median_img, pattern_size, corners2, ret)
      else:
        print("提取角点失败:",img_file)

      cv2.namedWindow("I", cv2.WINDOW_NORMAL)
      cv2.namedWindow("O", cv2.WINDOW_NORMAL)
      cv2.imshow("I", src_img)
      cv2.imshow("O", median_img)
      cv2.waitKey(0)
  cv2.destroyAllWindows()
if __name__ == "__main__":
  calib("E:\\data\\lingguang\\calib\\", 12)
  