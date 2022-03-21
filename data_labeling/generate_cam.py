import argparse
import logging
from imageio import imwrite, imsave
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import numpy as np
import os
import time
from realsense.camera import RealSenseCamera



logging.basicConfig(level=logging.INFO)

def on_click(event):
    date = str(time.strftime('%m%d%M', time.localtime(time.time())))
    imwrite(rgb_path + str(i) +'.png', rgb)
    imwrite(depth_path +str(i) + '.tif', depth)
    print('save')
    plt.close()


if __name__ == '__main__':
    path_name = ''
    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id='f1181780') #0003b661b825 # f0350818 # f1181780
    cam.connect() 
    date = str(time.strftime('%m%d%M', time.localtime(time.time())))
    save_path = './data_labeling/data/img/'+date + '_' + path_name +'/'
    rgb_path =  save_path + 'rgb/'
    depth_path = save_path + 'depth/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    rows = 2
    cols = 1
    i =1
    rgb = None
    depth = None

    while True:
        fig = plt.figure()
        image_bundle = cam.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']


        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(rgb)
        ax0.set_title('rgb_img')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(depth)
        ax0.set_title('depth_img')
        ax0.axis("off")
        
        plt.connect('button_press_event', on_click)

        axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
        
        bcut = Button(axcut, 'Capture', color='red', hovercolor='green')
        # plt.pause(0.1)
        plt.show()
        i += 1

    
        


       
        
