import cv2
from imageio import imread
import numpy as np


# img = imread('./test_img.png')
img = imread('./test.jpeg')

img_thresholded = cv2.inRange(img, (60, 60, 60), (140, 140, 140))

kernel = np.ones((10,10),np.uint8)
opening = cv2.morphologyEx(img_thresholded, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(len(contours))
i = 0

for contour in contours:
    (x,y),radius = cv2.minEnclosingCircle(contour)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(img,center,radius,(0,255,0),2)
    # labelling the circles around the centers, in no particular order.
    position = (center[0] - 10, center[1] + 10)
    text_color = (0, 0, 255)
    cv2.putText(img, str(i + 1), position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i +=1 
