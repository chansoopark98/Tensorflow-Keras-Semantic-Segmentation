#!/usr/bin/env python3
from audioop import mul
from glob import glob
import multiprocessing
from threading import Thread
from cv2 import HoughCircles
from models.model_builder import semantic_model, segmentation_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
from utils.pyrealsense_camera import RealSenseCamera
import rospy
# ros numpy install 방법 
# sudo apt-get install ros-$release-ros-numpy ($release는 현재 사용하는 ROS version (ex: melodic))
from std_msgs.msg import Float32, String
import message_filters
from sensor_msgs.msg import Image, CameraInfo
import json
from utils.CameraManager import CameraManager


tf.keras.backend.clear_session()
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

def load_mindVision():
    def jh_img_callback(img):
        global bgr
        rospy.get_rostime().to_sec()
        bgr = bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')

    color_sub = message_filters.Subscriber('mv_cam', Image) 
    color_sub.registerCallback(jh_img_callback)

    return bgr

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

parser = argparse.ArgumentParser()
parser.add_argument("--camera_mode",     type=int,   help="Camera Mode || 1 : RealSense launch  2 : MindVision", default=1)
args = parser.parse_args()
CAM_MODE = args.camera_mode
IMAGE_SIZE = (480, 640)

if CAM_MODE == 1:
    with open('./vision_grasp.json', 'r') as f_config:
        config_data = json.load(f_config)
    camera = 0
    camera_config_data = config_data["cameras"]
    for idx, cam in enumerate(camera_config_data):
        topic_info = cam["topics"]
        camera = CameraManager({'name':cam["name"],'cameraInfos':topic_info["info"],'colorStream':topic_info["color"],'depthStream':topic_info["depth"],'calibration':cam["calibration"]})
    camera.register_cb()

if __name__ == '__main__':
    # Segmentation 모델 불러오기
    
    model = semantic_model(image_size=(480, 640))

    # Segmentation 모델 가중치 불러오기
    weight_name = 'epoch200'
    model.load_weights(weight_name + '.h5')

    # Warm-Up Deep Learning Model's (한 번 Inference하여 대기 시간을 줄임) 
    roi_img = tf.zeros([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    roi_img = tf.cast(roi_img, dtype=tf.float32)
    roi_img = preprocess_input(roi_img, mode='torch')
    roi_img = tf.expand_dims(roi_img, axis=0)
    roi_pred = model.predict_on_batch(roi_img)

    # RosTopic으로 publish할 변수 (예측한 홀의 중심 x,y 좌표)
    previous_yx = [0 ,0]

    # Load Camera
    if CAM_MODE == 1:
        # Load RealSense Camera
        # camera = load_realSense()
        # Get Camera RGB shape (h, w, c)
        rgb_shape = camera.color.shape
        y_index = (rgb_shape[0] - IMAGE_SIZE[0]) // 2 
        x_index = (rgb_shape[1] - IMAGE_SIZE[1]) // 2



    elif CAM_MODE == 2:
        # Load MindVision Camera
        original = load_mindVision()

        # Get Camera RGB shape (h, w, c)
        rgb_shape = original.copy().shape
        y_index = (rgb_shape[0] - IMAGE_SIZE[0]) // 2 
        x_index = (rgb_shape[1] - IMAGE_SIZE[1]) // 2


    # Set x,y,w,h Variable Initialization
    x,y,w,h = 0, 0, 0, 0

    
    original = camera.color.copy()
    new_rgb = camera.color.copy()

  
    # # Main Loop
    while True:
        original = camera.color.copy()
        yx_coords = []
        x_list = []
        y_list = []
        rgb = original.copy()[y_index:y_index+IMAGE_SIZE[0], x_index:x_index+IMAGE_SIZE[1]]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # Inference                     
        img = tf.cast(rgb, dtype=tf.float32)
        img = preprocess_input(img, mode='torch')
        img = tf.expand_dims(img, axis=0)
        pred = model.predict_on_batch(img)
        pred = tf.math.argmax(pred, axis=-1)
        pred = tf.cast(pred, tf.uint8)
        
        result = pred[0]
        result = result.numpy()  * 127

        new_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        gray = original.copy()[y_index:y_index+IMAGE_SIZE[0], x_index:x_index+IMAGE_SIZE[1]]
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if np.any(result==254):
            mask = np.where(result == 254, 1, 0)
            argwhere = np.argwhere(mask == 1)
            max_coord = np.max(argwhere, axis=0)
            min_coord = np.min(argwhere, axis=0)

            y_min = min_coord[0]
            x_min = min_coord[1]
            y_max = max_coord[0]
            x_max = max_coord[1]

            mask[y_min:y_max, x_min:x_max] = 1

            gray *= mask.astype(np.uint8)
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


            draw_img = rgb.copy()
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
                x += x_index
                y += y_index
                if original.copy()[:, :, 0][y, x] != 0:
                    cv2.circle(original, (int(x), int(y)), int(3), (0, 0, 255), 3, cv2.LINE_AA)


        start = time.process_time()
        
        img_msg = bridge.cv2_to_imgmsg(new_rgb, encoding='bgr8')
        seg_result_pub.publish(img_msg)


        img_msg = bridge.cv2_to_imgmsg(original, encoding='bgr8')
        hole_result_pub.publish(img_msg)

        duration = (time.process_time() - start)
        print('time :', duration, 'sec')
   


        pub.publish(f'x : {previous_yx[0]} , y : {previous_yx[0]}')


