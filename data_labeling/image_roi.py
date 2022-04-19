import argparse
import time
import os
import glob
import cv2
import numpy as np
from imageio import imread, imwrite

parser = argparse.ArgumentParser()
parser.add_argument("--save_name",     type=str,   help="test result path", default='normal_')
parser.add_argument("--rgb_path",     type=str,   help="result path", default='./data/img/concent/original/041930_normal_tilt_/rgb/')
parser.add_argument("--depth_path",     type=str,   help="test  path", default='./data/img/concent/original/041930_normal_tilt_/depth/')
parser.add_argument("--result_path",     type=str,   help="test result ", default='./data/img/concent/roi/041930_normal_tilt_/result/')

args = parser.parse_args()
SAVE_NAME = args.save_name
RGB_PATH = args.rgb_path
DEPTH_PATH = args.depth_path
RESULT_PATH = args.result_path

SAVE_RGB_PATH = RESULT_PATH + 'rgb/'
SAVE_DEPTH_PATH = RESULT_PATH + 'depth/'
os.makedirs(SAVE_RGB_PATH, exist_ok=True)
os.makedirs(SAVE_DEPTH_PATH, exist_ok=True)

img_list = glob.glob(os.path.join(RGB_PATH + '*.png'))
img_list.sort()

depth_list = glob.glob(os.path.join(DEPTH_PATH + '*.tif'))
depth_list.sort()

print(img_list)
sample_rgb = cv2.imread(img_list[0])


# x, y, w, h = cv2.selectROI(sample_rgb)

# print(x, y, w, h) #411 201 1000 1000
x, y, w, h = 411, 201, 1024, 1024
cv2.destroyAllWindows()

for i in range(len(img_list)):
    rgb = cv2.imread(img_list[i])
    depth = imread(depth_list[i])
    roi_rgb = rgb.copy()[y:y+h, x:x+w]
    roi_depth = depth.copy()[y:y+h, x:x+w]

    cv2.imwrite(SAVE_RGB_PATH +  SAVE_NAME  + 'rgb_' + str(i+1) +'.png', roi_rgb)
    imwrite(SAVE_DEPTH_PATH + SAVE_NAME + 'depth_' + str(i+1) +'.tif', roi_depth)

