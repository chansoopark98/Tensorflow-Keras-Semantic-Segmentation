#!/usr/bin/env python3
from glob import glob
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
from utils.CameraManager import CameraBuilder

tf.keras.backend.clear_session()
bridge = CvBridge()

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
        camera = CameraBuilder({'name':cam["name"],'cameraInfos':topic_info["info"],'colorStream':topic_info["color"],'depthStream':topic_info["depth"],'calibration':cam["calibration"]})
    camera.register_cb()

if __name__ == '__main__':
    # Segmentation 모델 불러오기
    
    model = segmentation_model(image_size=(480, 640))

    # Segmentation 모델 가중치 불러오기
    weight_name = 'semantic_full_weight'
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

    # Main Loop
    while True:
        if CAM_MODE==1:
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

        # start = time.process_time()
        # duration = (time.process_time() - start)
        # print('cv :', duration, 'sec')
        new_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        

        img_msg = bridge.cv2_to_imgmsg(new_rgb, encoding='bgr8')
        seg_result_pub.publish(img_msg)

        
        hole_result = np.where(result== 254,254, 0)
        hole_result = hole_result.astype(np.uint8)
        contours, _ = cv2.findContours(hole_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangle_contours = []
        if len(contours) != 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= 20:
                    rectangle_contours.append(contour)

            if len(rectangle_contours) != 0:
                area_mask = hole_result.copy()
                for contours_idx in range(len(rectangle_contours)):
                    x,y,w,h = cv2.boundingRect(rectangle_contours[contours_idx])
                    semantic_area = area_mask[y:y+h, x:x+w]
                    
                    # coord = np.mean(np.argwhere(semantic_area == 254), axis=0)
                    
                    argwhere = np.argwhere(semantic_area == 254)
                    max_coord = np.max(argwhere, axis=0)
                    min_coord = np.min(argwhere, axis=0)

                    coord = (max_coord - min_coord) //2
                    
                    
                    yx_coords.append(coord)
                    x_list.append(x)
                    y_list.append(y)

                
                # cropped_input_img = result.copy()[y:y+h, x:x+w]
                
                previous_yx = []
                
                for i in range(len(yx_coords)):
                    if np.isnan(yx_coords[i][0]) != True: 
                        previous_yx.append([yx_coords[i][0] + y_list[i], yx_coords[i][1] + x_list[i]])
                
                original[y_index:y_index+IMAGE_SIZE[0], x_index:x_index+IMAGE_SIZE[1]] = rgb
                for i in range(len(previous_yx)):
                    
                        # cv2.circle(rgb, (int(yx_coords[1]), int(yx_coords[0])), int(3), (0, 0, 255), 3, cv2.LINE_AA)

                    cv2.circle(original, (int(previous_yx[i][1]+x_index), int(previous_yx[i][0])+y_index), int(3), (0, 0, 255), 3, cv2.LINE_AA)
            
        img_msg = bridge.cv2_to_imgmsg(original, encoding='bgr8')
        hole_result_pub.publish(img_msg)
        pub.publish(f'x : {previous_yx[0]} , y : {previous_yx[0]}')

