import argparse
import time
import os
import glob
import cv2
import numpy as np

def check_boolean(value):
    if value == 1:
        return True
    else:
        return False

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="test result path", default='./data_labeling/data/img/0408/040829_exposure_1000_gain_100_25cm_gray3/result/semantic_label_mask_result/semantic_mask/input')
parser.add_argument("--mask_path",     type=str,   help="test result path", default='./data_labeling/data/img/0408/040829_exposure_1000_gain_100_25cm_gray3/result/semantic_label_mask_result/semantic_mask/gt')



args = parser.parse_args()
RGB_PATH = args.rgb_path
MASK_PATH = args.mask_path
batch_idx = 0
avg_duration = 0
ng_time = 0

img_list = glob.glob(os.path.join(RGB_PATH,'*.png'))
img_list.sort()

mask_list = glob.glob(os.path.join(MASK_PATH,'*.png'))
mask_list.sort()

for i in range(len(img_list)):
    
    img = cv2.imread(img_list[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = cv2.imread(mask_list[i])
    mask = np.where(mask==2, 1, 0)
    binary_mask = np.expand_dims(mask[:, :, 0].astype(np.uint8), axis=-1)

    img *= binary_mask

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
                cv2.circle(draw_img, (int(x), int(y)), int(3), (0, 0, 255), 3, cv2.LINE_AA)
        # # 키 포인트를 빨간색으로 표시 ---③
        # draw_img = cv2.drawKeypoints(img.copy(), keypoints, None, (0,0,255),
        #                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow("Trackbar Windows", draw_img)
    

# print(f"avg inference time : {(avg_duration / len(img_list)) // 1000000}ms.")
# print(f"No good images : {ng_time}.")


