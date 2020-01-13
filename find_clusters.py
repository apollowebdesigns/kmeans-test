import cv2
import numpy as np

img = cv2.imread('test.png', 0)
output_img = img.copy()
cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)

kernel = np.ones((50, 70), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)

contours, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a white rectangle to visualize the bounding rect
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

cv2.imwrite('output.png', output_img)
cv2.waitKey(0)
