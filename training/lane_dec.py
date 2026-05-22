import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.lane_utils import fit_polynomial, single_fit


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python lane_dec.py <image_path>")
        sys.exit(1)

    img_org = cv2.imread(sys.argv[1], 0)
    print(img_org.shape)

    a, img = cv2.threshold(img_org, 75, 255, cv2.THRESH_BINARY_INV)
    img = cv2.erode(img, kernel=(4, 4), iterations=4)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=(5, 5))
    height, width = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[(0, height * 0.4), (width, height * 0.4), (width, height), (0, height)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)

    edge = 145
    double_hit = single_hit = 0
    left_x, left_y, right_x, right_y = [], [], [], []
    single_x, single_y = [], []
    prediction = -1

    cv2.imshow('win', masked_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    while edge < 240:
        temp = masked_image[edge:edge + 5]
        contour, _ = cv2.findContours(temp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contour if 10 < cv2.contourArea(c) < 60]
        le = len(contours)
        cent = []

        if le == 0:
            edge += 5
            continue
        elif le == 1:
            x, y, w, h = cv2.boundingRect(contours[0])
            cent.append([x, y + edge])
            cv2.circle(img_org, (x, y + edge), 5, (255), -1)
            if double_hit < 10:
                single_x.append(cent[0][0])
                single_y.append(cent[0][1])
                single_hit += 1
        elif le == 2:
            for cont in contours:
                x, y, w, h = cv2.boundingRect(cont)
                cent.append([x, y + edge])
                cv2.circle(img_org, (x, y + edge), 5, (255), -1)
            if cent[0][0] > cent[1][0]:
                left_x.append(cent[0][0]); left_y.append(cent[0][1])
                right_x.append(cent[1][0]); right_y.append(cent[1][1])
            else:
                left_x.append(cent[1][0]); left_y.append(cent[1][1])
                right_x.append(cent[0][0]); right_y.append(cent[0][1])
            double_hit += 1
        else:
            pts, abso = [], []
            for cont in contours:
                x, y, w, h = cv2.boundingRect(cont)
                pts.append([x + int(w / 2), y + edge])
                abso.append(abs(x + int(w / 2) - 210))
            ins = np.argsort(abso)
            pts = np.array(pts)[ins]
            cent = [pts[0], pts[1]]
            if cent[0][0] > cent[1][0]:
                left_x.append(cent[0][0]); left_y.append(cent[0][1])
                right_x.append(cent[1][0]); right_y.append(cent[1][1])
            else:
                left_x.append(cent[1][0]); left_y.append(cent[1][1])
                right_x.append(cent[0][0]); right_y.append(cent[0][1])
            double_hit += 1
        edge += 5

    print(double_hit, single_hit, "------------")
    emp = np.zeros_like(img_org)

    if len(single_x) > 2 or len(left_x) > 2:
        if double_hit > 8:
            plot_y, left_fit, left_fit_x, right_fit, right_fit_x = fit_polynomial(emp, left_x, left_y, right_x, right_y)
            plt.imshow(img_org, cmap='gray')
            print(left_fit)
            print(right_fit)
            print('double lane')
            current = len(left_x) - 1
            curr_centre_x = (left_x[current] + right_x[current]) // 2
            curr_centre_y = (left_y[current] + right_y[current]) // 2
            plt.plot(left_fit_x, plot_y, color='red')
            plt.plot(right_fit_x, plot_y, color='red')
            plt.xlabel('Lane_Detected_Polynomial_')
            plt.show()

            total_deg, total_rad = 0.0, 0.0
            for i in range(len(left_x) - 1):
                cx = (left_x[i] + right_x[i]) // 2
                cy = (left_y[i] + right_y[i]) // 2
                b2 = float(abs(cx - curr_centre_x))
                h2 = float(abs(cy - curr_centre_y))
                rad = math.atan(h2 / b2) if b2 != 0.0 else 1.5708
                total_rad += rad
                total_deg += abs(90 - rad * (180.0 / math.pi))

            degree_comp = total_deg / max(current, 1)
            rad_comp = total_rad / max(current, 1)
            sl = math.tan(rad_comp)
            cx = 145
            cy = int(sl * (cx - curr_centre_x) + curr_centre_y)
            cv2.line(img_org, (cx, cy), (curr_centre_x, curr_centre_y), (255), 3)

            if degree_comp > 55:
                prediction = 3
            elif degree_comp > 35:
                prediction = 1 if curr_centre_x > cx else 2
            else:
                prediction = 0
            if cx > 390 or cx < 30:
                prediction = 3

        elif single_hit > 4:
            plot_y, fit, fit_x = single_fit(emp, single_x, single_y)
            plt.imshow(img_org, cmap='gray')
            plt.plot(fit_x, plot_y, color='red')
            plt.xlabel('Lane_Detected_Polynomial_')
            plt.show()
            print('single lane', fit)
            current = len(single_x) - 1
            curr_x = single_x[current]
            first_x = single_x[0]
            if curr_x < first_x:
                prediction = 2
            elif curr_x > first_x:
                prediction = 1
            else:
                prediction = 0
        else:
            print('No lane found')
    else:
        print('No lane found')

    print('Prediction:', prediction)
    cv2.imshow("image", img_org)
    cv2.waitKey()
    cv2.destroyAllWindows()
