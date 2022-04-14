#!/usr/bin/env python3
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
import rospy
# ros numpy install 방법 
# sudo apt-get install ros-$release-ros-numpy ($release는 현재 사용하는 ROS version (ex: melodic))
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image, CameraInfo
import json
from utils.CameraManager import CameraManager


bridge = CvBridge()

def check_boolean(value):
    if value == 1:
        return True
    else:
        return False

rospy.init_node('topic_publisher', anonymous=True)
pub = rospy.Publisher('counter', String, queue_size=1)
seg_result_pub = rospy.Publisher('Segmentation_result', Image, queue_size=1)
hole_result_pub = rospy.Publisher('Final_result', Image, queue_size=1)
rate = rospy.Rate(60)

def hole_image_pub(pub, pub_img):
    while True:
        try:
                img_msg = bridge.cv2_to_imgmsg(pub_img, encoding='bgr8')
                pub.publish(img_msg)
        except CvBridgeError as e:
                print(e)

def seg_image_pub(pub, pub_img):
    try:
            img_msg = bridge.cv2_to_imgmsg(pub_img, encoding='bgr8')
            pub.publish(img_msg)
    except CvBridgeError as e:
            print(e)

def waitCamera(cam):
    while True:
        try:
            if cam.color.copy() is not None:
                break
        except:
            print('wait for camera')

parser = argparse.ArgumentParser()
parser.add_argument("--camera_mode",     type=int,   help="Camera Mode || 1 : RealSense launch  2 : MindVision", default=1)
args = parser.parse_args()
CAM_MODE = args.camera_mode
IMAGE_SIZE = (480, 640)

if CAM_MODE == 1:
    with open('./camera_infos.json', 'r') as f_config:
        config_data = json.load(f_config)
    camera = 0
    camera_config_data = config_data["cameras"]
    for idx, cam in enumerate(camera_config_data):
        topic_info = cam["topics"]
        camera = CameraManager({'name':cam["name"],'cameraInfos':topic_info["info"],'colorStream':topic_info["color"],'depthStream':topic_info["depth"],'calibration':cam["calibration"]})
    camera.register_cb()

if __name__ == '__main__':
    waitCamera(camera)
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
    cv2.setTrackbarPos("minArea", "Trackbar Windows", 30)

    cv2.setTrackbarPos("filterByCircularity", "Trackbar Windows", 1)
    cv2.setTrackbarPos("minCircularity", "Trackbar Windows", 1)

    cv2.setTrackbarPos("filterByConvexity", "Trackbar Windows", 1)
    cv2.setTrackbarPos("minConvexity", "Trackbar Windows", 1)

    cv2.setTrackbarPos("filterByInertia", "Trackbar Windows", 1)
    cv2.setTrackbarPos("minInertiaRatio", "Trackbar Windows", 1)

    
    while cv2.waitKey(1) != ord('q'):
        draw_img = camera.color.copy()
        gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)

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
        # Filter by Area
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

        # SimpleBlobDetector 생성
        detector = cv2.SimpleBlobDetector_create(params) # SimpleBlobDetector
        # 키 포인트 검출
        keypoints = detector.detect(gray)
        
        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = int(keypoint.size)
            cv2.circle(draw_img, (int(x), int(y)), s, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Trackbar Windows", draw_img)
    cv2.destroyAllWindows()
    print('before_keypoint')
    # Load Camera
    if CAM_MODE == 1:
        # Load RealSense Camera
        # camera = load_realSense()
        # Get Camera RGB shape (h, w, c)
        rgb_shape = camera.color.shape
        y_index = (rgb_shape[0] - IMAGE_SIZE[0]) // 2 
        x_index = (rgb_shape[1] - IMAGE_SIZE[1]) // 2

    original = camera.color.copy()
    new_rgb = camera.color.copy()

     # Main Loop
    while True:
        start = time.process_time()
        original = camera.color.copy()
        rgb = original.copy()[y_index:y_index+IMAGE_SIZE[0], x_index:x_index+IMAGE_SIZE[1]]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
               
        gray = original.copy()[y_index:y_index+IMAGE_SIZE[0], x_index:x_index+IMAGE_SIZE[1]]
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        params = cv2.SimpleBlobDetector_Params()


        # Change thresholds
        params.minThreshold = minThreshold
        params.maxThreshold = maxThreshold
        # Filter by Area
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

        # SimpleBlobDetector 생성 
        detector = cv2.SimpleBlobDetector_create(params) # SimpleBlobDetector
        # 키 포인트 검출
        keypoints = detector.detect(gray)
        
        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = int(keypoint.size)
            x += x_index
            y += y_index
            if original.copy()[:, :, 0][y, x] != 0:
                cv2.circle(original, (x, y), s, (0, 0, 255), 1, cv2.LINE_AA)

        img_msg = bridge.cv2_to_imgmsg(new_rgb, encoding='bgr8')
        seg_result_pub.publish(img_msg)

        img_msg = bridge.cv2_to_imgmsg(original, encoding='bgr8')
        hole_result_pub.publish(img_msg)

        duration = (time.process_time() - start)
        print('time :', duration, 'sec')
   
        pub.publish(f'x : {x} , y : {y}')


