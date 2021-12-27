import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("../data/paowuxian.jpg")

    KF = cv2.KalmanFilter(4, 2, 0)
    KF.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    KF.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    KF.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
    KF.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.1
    last_measurement = current_measurement = np.array((2, 1), np.float32)

    last_predicition = current_prediction = np.array([[2], [2]], np.float32)

    points = [
              (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
              (10, 10),
              (20, 20),
              (45, 45),
              (55, 55),
              (60, 60),
        (65, 65),
        (68, 68),
        (70, 70),
        (75, 75),
        (80, 80),
        (85, 90),
        (90, 90),
        (100, 100),
        (105, 102),
        (108, 105),
        (110, 110),
        (120, 120),
        (200, 300),
        (210, 250)]

    for i in range(len(points)):
        last_measurement = current_measurement
        last_prediction = current_prediction

        current_measurement = np.array([[np.float32(points[i][0])],[np.float32(points[i][1])]])
        KF.correct(current_measurement)

        current_prediction = KF.predict()

        lmx, lmy = last_measurement[0], last_measurement[1]
        cmx, cmy = current_measurement[0], current_measurement[1]

        lpx, lpy = last_prediction[0], last_prediction[1]
        cpx, cpy = current_prediction[0], current_prediction[1]

        cv2.line(img, (int(lmx), int(lmy)), (int(cmx), int(cmy)), (0,255,0), 1)
        cv2.line(img, (int(lpx), int(lpy)), (int(cpx), int(cpy)), (0,0,255), 1)
        cv2.imshow("test", img)
        cv2.waitKey(0)

