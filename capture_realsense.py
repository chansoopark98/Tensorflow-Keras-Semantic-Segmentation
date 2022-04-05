#!/usr/bin/env python3
import argparse
import time
import cv2
import json

from cv2 import imwrite
from utils.CameraManager import CameraManager
import os


parser = argparse.ArgumentParser()
parser.add_argument("--camera_mode",     type=int,   help="Camera Mode || 1 : RealSense launch  2 : MindVision", default=1)
args = parser.parse_args()
CAM_MODE = args.camera_mode
IMAGE_SIZE = (480, 640)
ROI_SIZE = (128, 128)

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

    path_name = 'gray_30cm'
    # Connect to Camera
    print('Connecting to camera...')
    date = str(time.strftime('%m%d%M', time.localtime(time.time())))
    save_path = './data_labeling/data/img/'+date + '_' + path_name +'/'
    rgb_path =  save_path + 'rgb/'
    depth_path = save_path + 'depth/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    i=0

    # Main Loop 
    while True:
        
        original = camera.color.copy()
        cv2.imshow("Image", original)
        key = cv2.waitKey(0)

        if key == ord('1'):
            imwrite(rgb_path + path_name + str(i) +'.png', original)
            i += 1
        elif key == ord('q'):
            break
        
