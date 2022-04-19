import argparse
import time
import os
import glob
import cv2
import numpy as np

# https://github.com/hfutcgncas/normalSpeed

def check_boolean(value):
    if value == 1:
        return True
    else:
        return False

parser = argparse.ArgumentParser()
parser.add_argument("--result_path",     type=str,   help="test result path", default='./test_imgs/hand/')
args = parser.parse_args()
RESULT_PATH = args.result_path


img_list = glob.glob(os.path.join(RESULT_PATH,'*.jpg'))
img_list.sort()


for i in range(len(img_list)):
    
    img = cv2.imread(img_list[i])
    img = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set nameWindow
    cv2.namedWindow("Trackbar Windows")
    
    # Set createTrackbars
    cv2.createTrackbar("minThreshold", "Trackbar Windows", 1, 255, lambda x : x)
    cv2.createTrackbar("maxThreshold", "Trackbar Windows", 1, 255, lambda x : x)

    cv2.createTrackbar("filterByArea", "Trackbar Windows", 0, 1, lambda x : x)
    cv2.createTrackbar("minArea", "Trackbar Windows", 1, 1500, lambda x : x)

    cv2.createTrackbar("filterByCircularity", "Trackbar Windows", 0, 1, lambda x : x)
    cv2.createTrackbar("minCircularity", "Trackbar Windows", 1, 10, lambda x : x)

    cv2.createTrackbar("filterByConvexity", "Trackbar Windows", 0, 1, lambda x : x)
    cv2.createTrackbar("minConvexity", "Trackbar Windows", 1, 20, lambda x : x)

    cv2.createTrackbar("filterByInertia", "Trackbar Windows", 0, 1, lambda x : x)
    cv2.createTrackbar("minInertiaRatio", "Trackbar Windows", 1, 10, lambda x : x)


    # Set default value
    cv2.setTrackbarPos("minThreshold", "Trackbar Windows", 10)
    cv2.setTrackbarPos("maxThreshold", "Trackbar Windows", 200)

    cv2.setTrackbarPos("filterByArea", "Trackbar Windows", 1)
    cv2.setTrackbarPos("minArea", "Trackbar Windows", 1500)

    cv2.setTrackbarPos("filterByCircularity", "Trackbar Windows", 1)
    cv2.setTrackbarPos("minCircularity", "Trackbar Windows", 1)

    cv2.setTrackbarPos("filterByConvexity", "Trackbar Windows", 1)
    cv2.setTrackbarPos("minConvexity", "Trackbar Windows", 1)

    cv2.setTrackbarPos("filterByInertia", "Trackbar Windows", 1)
    cv2.setTrackbarPos("minInertiaRatio", "Trackbar Windows", 1)


    while cv2.waitKey(1) != ord('q'):
        draw_img = img.copy()
        minThreshold = cv2.getTrackbarPos("minThreshold", "Trackbar Windows")
        maxThreshold = cv2.getTrackbarPos("maxThreshold", "Trackbar Windows")

        filterByArea = cv2.getTrackbarPos("filterByArea", "Trackbar Windows")
        minArea = cv2.getTrackbarPos("minArea", "Trackbar Windows")

        filterByCircularity =cv2.getTrackbarPos("filterByCircularity", "Trackbar Windows")
        minCircularity = cv2.getTrackbarPos("minCircularity", "Trackbar Windows")

        filterByConvexity = cv2.getTrackbarPos("filterByConvexity", "Trackbar Windows")
        minConvexity = cv2.getTrackbarPos("minConvexity", "Trackbar Windows")

        filterByInertia = cv2.getTrackbarPos("filterByInertia", "Trackbar Windows")
        minInertiaRatio = cv2.getTrackbarPos("minInertiaRatio", "Trackbar Windows")

        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = minThreshold
        params.maxThreshold = maxThreshold

        # Filter by Area.
        params.filterByArea = check_boolean(filterByArea)
        params.minArea = minArea
        # Filter by Circularity
        params.filterByCircularity = check_boolean(filterByCircularity)        
        params.minCircularity = minCircularity * 0.1
        # Filter by Convexity
        params.filterByConvexity = check_boolean(filterByConvexity)      
        params.minConvexity = minConvexity * 0.1
        
        # Filter by Inertia
        params.filterByInertia = check_boolean(filterByInertia)   
        params.minInertiaRatio = minInertiaRatio * 0.1


        # SimpleBlobDetector 생성 ---①
        detector = cv2.SimpleBlobDetector_create(params) # SimpleBlobDetector
        # 키 포인트 검출 ---②
        keypoints = detector.detect(gray)
        
        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = keypoint.size
            print(img.copy()[y, x])
            if img.copy()[:, :, 0][y, x] != 0:
                cv2.circle(draw_img, (int(x), int(y)), int(s), (0, 0, 255), 1, cv2.LINE_AA)
        # # 키 포인트를 빨간색으로 표시 ---③
        # draw_img = cv2.drawKeypoints(img.copy(), keypoints, None, (0,0,255),
        #                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow("Trackbar Windows", draw_img)
    

# print(f"avg inference time : {(avg_duration / len(img_list)) // 1000000}ms.")
# print(f"No good images : {ng_time}.")


