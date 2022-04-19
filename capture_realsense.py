#!/usr/bin/env python3
import time
import cv2
import json
from cv2 import imwrite
from utils.CameraManager import CameraBuilder
import os
import numpy as np

with open('./camera_infos.json', 'r') as f_config:
    config_data = json.load(f_config)
camera = 0
camera_config_data = config_data["cameras"]
for idx, cam in enumerate(camera_config_data):
    topic_info = cam["topics"]
    camera = CameraBuilder({
        'name':'camera01',
        'cameraInfos':topic_info["info"],
        'colorStream':'/rgb/image_raw',
        'depthStream':'/depth_to_rgb/image_raw',
        'calibration':cam["calibration"]
    })
    # camera = CameraBuilder({'name':cam["name"],'cameraInfos':topic_info["info"],'colorStream':topic_info["color"],'depthStream':topic_info["depth"],'calibration':cam["calibration"]})
camera.register_cb()

if __name__ == '__main__':
    path_name = 'normal_tilt_'
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
        try:
            
            if camera.color is not None:
                
                break
        except:
            continue

    while True:
        try:
            
            if camera.depth is not None:
                
                break
        except:
            continue


    while True:
        original = camera.color.copy()
        depth = camera.depth.copy()
        print(depth.shape)
        copy_img = original.astype(np.uint8).copy()
        copy_img = cv2.resize(copy_img, dsize=(1280, 720))
        cv2.imshow("Image", copy_img)
        key = cv2.waitKey(0)

        if key == ord('1'):
            imwrite(rgb_path + path_name + str(i) +'.png', original)
            imwrite(depth_path + path_name + str(i) +'.tif', depth)
            i += 1
        elif key == ord('q'):
            break
        

