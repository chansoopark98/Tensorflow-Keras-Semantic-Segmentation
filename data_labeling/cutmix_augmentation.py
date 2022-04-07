import numpy as np
import cv2
from imageio import imread, imwrite
import glob
import os
import argparse
import random
from utils import load_imgs, canny_edge, find_contours
import natsort
import re 
import math
from pathlib import Path
from tensorflow.keras.preprocessing.image import random_rotation

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040533_24cm_white_b1/result/rgb/')
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040533_24cm_white_b1/result/mask/')
parser.add_argument("--depth_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040533_24cm_white_b1/result/depth/')
parser.add_argument("--result_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040533_24cm_white_b1/result/augmentation/')

parser.add_argument("--bg_path",     type=str,   help="raw image path", default='./data_labeling/data/img/dtd/images/')


args = parser.parse_args()
RGB_PATH = args.rgb_path
MASK_PATH = args.mask_path
BG_PATH = args.bg_path
# OPEN_DEPTH_PATH = args.depth_path
OUTPUT_PATH = args.result_path
RESOLUTION = (640,480)

os.makedirs(OUTPUT_PATH, exist_ok=True)

OUT_RGB_PATH = OUTPUT_PATH + 'rgb/'
OUT_DEPTH_PATH = OUTPUT_PATH + 'depth/'
OUT_MASK_PATH = OUTPUT_PATH + 'mask/'
os.makedirs(OUT_RGB_PATH, exist_ok=True)
os.makedirs(OUT_DEPTH_PATH, exist_ok=True)
os.makedirs(OUT_MASK_PATH, exist_ok=True)

rgb_list = glob.glob(os.path.join(RGB_PATH+'*.png'))
rgb_list = natsort.natsorted(rgb_list,reverse=True)

mask_list = glob.glob(os.path.join(MASK_PATH+'*.png'))
mask_list = natsort.natsorted(mask_list,reverse=True)

bg_list =  glob.glob(os.path.join(BG_PATH,  '*', '*.jpg'))
bg_list = natsort.natsorted(bg_list,reverse=True)

def cutmix(img, mask, bg, name):
    bg = cv2.resize(bg, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)

    # mask = mask[:, :, 0]
    binary_mask = np.where(mask >= 200, 1, 0).astype(np.uint8)
    
    img *= binary_mask


    bg_background = bg * np.where(binary_mask == 1, 0, 1).astype(np.uint8)
    bg_background += img

    cv2.imshow('test', bg_background)
    cv2.waitKey(0)
    
    


if __name__ == '__main__': 
    print(rgb_list[0], mask_list[0])
    i = 1
    for idx in range(len(rgb_list)):
        img = cv2.imread(rgb_list[idx])
        mask = cv2.imread(mask_list[idx])
        bg_idx = random.choice(range(len(bg_list)))
        bg = cv2.imread(bg_list[bg_idx])
        # depth = imread(depth_list[idx])
        
        
        name = rgb_list[idx].split('/')[4] + '_' + str(i)

        i+=1

        cutmix(img, mask, bg, name)
        # blur

        # shift

        # rotate

        # # img_name = rgb_files[idx].split('/')[5].split('\\')[1].split('.')[0]
        
        
        

        
        
