import argparse
import os
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--result_path",     type=str,   help="test result path", default='./test_imgs/')
args = parser.parse_args()
RESULT_PATH = args.result_path


img_list = glob.glob(os.path.join(RESULT_PATH,'*.png'))
img_list.sort()

def onChange(pos):
    pass

cv2.namedWindow("Trackbar Windows")

cv2.createTrackbar("minDistance", "Trackbar Windows", 0, 255, lambda x : x)
cv2.createTrackbar("CannyThreshold", "Trackbar Windows", 0, 255, lambda x : x)
cv2.createTrackbar("CenterThreshold", "Trackbar Windows", 0, 255, lambda x : x)
cv2.createTrackbar("minRadius", "Trackbar Windows", 0, 255, lambda x : x)
cv2.createTrackbar("maxRadius", "Trackbar Windows", 0, 255, lambda x : x)

cv2.setTrackbarPos("minDistance", "Trackbar Windows", 2)
cv2.setTrackbarPos("CannyThreshold", "Trackbar Windows", 15)
cv2.setTrackbarPos("CenterThreshold", "Trackbar Windows", 17)
cv2.setTrackbarPos("minRadius", "Trackbar Windows", 2)
cv2.setTrackbarPos("maxRadius", "Trackbar Windows", 10)


for i in range(len(img_list)):
    
    img = cv2.imread(img_list[i])
    original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    

    

    while cv2.waitKey(1) != ord('q'):
        img = original.copy()
        minDistance = cv2.getTrackbarPos("minDistance", "Trackbar Windows")
        CannyThreshold = cv2.getTrackbarPos("CannyThreshold", "Trackbar Windows")
        CenterThreshold = cv2.getTrackbarPos("CenterThreshold", "Trackbar Windows")
        minRadius = cv2.getTrackbarPos("minRadius", "Trackbar Windows")
        maxRadius = cv2.getTrackbarPos("maxRadius", "Trackbar Windows")
        
    

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDistance,
                param1=CannyThreshold, param2=CenterThreshold, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            for i in circles[0]:
                cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), 255, 1)
                cv2.circle(img, (int(i[0]), int(i[1])), 0, 0, -1)
        cv2.imshow("Trackbar Windows", img)
        