"""
Experimental patch-based lane labelling tool.

NOTE: This script produces variable-width image patches and saves them to
lane_data.npz.  That file has shape (n,) not (n, 50400), so load_data()
will always skip it.  This tool is NOT part of the active training pipeline
— use collect_training.py instead.

Usage:
    python lane_detec_neural.py <image_path>
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python lane_detec_neural.py <image_path>")
    sys.exit(1)

img_org = cv2.imread(sys.argv[1], 0)
if img_org is None:
    print("Could not read image:", sys.argv[1])
    sys.exit(1)

plt.imshow(img_org, cmap='gray')
plt.show()

_, img = cv2.threshold(img_org, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=(4, 4))
height, width = img.shape
mask = np.zeros_like(img)

polygon = np.array([[(0, height * 0.1), (width, height * 0.1), (width, height), (0, height)]], np.int32)
cv2.fillPoly(mask, polygon, 255)
masked_image = cv2.bitwise_and(img, mask)
masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel=(5, 5))
masked_image = cv2.dilate(masked_image, kernel=(10, 10), iterations=10)
cv2.imshow('win', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

edge = 0
training_set = []
training_label = []

while edge < img_org.shape[0]:
    temp = masked_image[edge:edge + 40]
    contours, _ = cv2.findContours(temp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV 4 returns 2 values
    for i, contour in enumerate(contours):
        disp_img = img_org.copy()
        temp_img = img_org[edge:edge + 40].copy()
        if cv2.contourArea(contour) < 1000:
            cv2.drawContours(temp_img, contours, i, 255, -1)
            x, y, w, h = cv2.boundingRect(contour)
            cent_x = x + w // 2
            store = img_org[edge:edge + 40, cent_x - 20:cent_x + 20]
            cv2.rectangle(temp_img, (cent_x - 20, 0), (cent_x + 20, 39), 255, 1)
            disp_img[edge:edge + 40] = temp_img
            cv2.imshow('win', disp_img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('y'):
                training_set.append(store.flatten())
                training_label.append([1, 0])
            else:
                training_set.append(store.flatten())
                training_label.append([0, 1])
            print(w)
    edge += 40

print(len(training_set))
np.savez('../data/lane_data.npz', train=np.array(training_set), train_labels=np.array(training_label))
