import argparse
import os
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--result_path",     type=str,   help="test result path", default='./test_imgs/hand/')
args = parser.parse_args()
RESULT_PATH = args.result_path


img_list = glob.glob(os.path.join(RESULT_PATH,'*.png'))
img_list.sort()

def onChange(pos):
    pass

for i in range(len(img_list)):
    
    img = cv2.imread(img_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Trackbar Windows")
    cv2.createTrackbar("minValue", "Trackbar Windows", 0, 255, lambda x : x)
    cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)

    cv2.setTrackbarPos("minValue", "Trackbar Windows", 127)
    cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)

    while cv2.waitKey(1) != ord('q'):
        thresh = cv2.getTrackbarPos("minValue", "Trackbar Windows")
        maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")
        
        canny = cv2.Canny(img, thresh, maxval)
        cv2.imshow("Trackbar Windows", canny)

        