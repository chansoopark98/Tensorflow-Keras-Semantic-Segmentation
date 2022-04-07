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
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040523_24cm_gray_d3_noExposure/result/semantic_label_mask_result/roi_mask/input/')
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040523_24cm_gray_d3_noExposure/result/semantic_label_mask_result/roi_mask/gt/')
parser.add_argument("--result_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040523_24cm_gray_d3_noExposure/result/semantic_label_mask_result/roi_mask/augmentation/')

parser.add_argument("--bg_path",     type=str,   help="raw image path", default='./data_labeling/data/img/dtd/images/')

args = parser.parse_args()
RGB_PATH = args.rgb_path
MASK_PATH = args.mask_path
BG_PATH = args.bg_path
OUTPUT_PATH = args.result_path


os.makedirs(OUTPUT_PATH, exist_ok=True)

OUT_RGB_PATH = OUTPUT_PATH + 'rgb/'

OUT_MASK_PATH = OUTPUT_PATH + 'mask/'
os.makedirs(OUT_RGB_PATH, exist_ok=True)

os.makedirs(OUT_MASK_PATH, exist_ok=True)

rgb_list = glob.glob(os.path.join(RGB_PATH+'*.png'))
rgb_list = natsort.natsorted(rgb_list,reverse=True)

mask_list = glob.glob(os.path.join(MASK_PATH+'*.png'))
mask_list = natsort.natsorted(mask_list,reverse=True)

bg_list =  glob.glob(os.path.join(BG_PATH,  '*', '*.jpg'))
bg_list = natsort.natsorted(bg_list,reverse=True)

def img_shift(rgb, mask, name):
    rand_x = random.randint(0, 100)
    rand_y = random.randint(0, 100)

    shift_x = rand_x
    shift_y = rand_y
    h, w = rgb.shape[:2]

    # plus affine
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aff_rgb_p = cv2.warpAffine(rgb, M, (w, h))
    # aff_depth_p = cv2.warpAffine(depth, M, (w, h))
    aff_mask_p = cv2.warpAffine(mask, M, (w, h))

    cv2.imwrite(OUT_RGB_PATH + name + '_p_shift' +'_.png', aff_rgb_p)
    # imwrite(OUT_DEPTH_PATH + name + '_p_shift'+'_depth.tif', aff_depth_p)
    cv2.imwrite(OUT_MASK_PATH +name + '_p_shift' +'_mask.png', aff_mask_p)

    M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    aff_rgb_m = cv2.warpAffine(rgb, M, (w, h))
    # aff_depth_m = cv2.warpAffine(depth, M, (w, h))
    aff_mask_m = cv2.warpAffine(mask, M, (w, h))

    cv2.imwrite(OUT_RGB_PATH + name + '_m_shift' +'_.png', aff_rgb_m)
    # imwrite(OUT_DEPTH_PATH + name + '_m_shift'+'_depth.tif', aff_depth_m)
    cv2.imwrite(OUT_MASK_PATH +name + '_m_shift' +'_mask.png', aff_mask_m)

    print('save_shift')
    # minus affine

def img_blur(img, mask, name):
    k = random.randrange(3,21,2)
    img = cv2.GaussianBlur(img, (k,k), 0)
    # depth = cv2.GaussianBlur(depth, (k,k), 0)
    # mask = cv2.GaussianBlur(mask, (k,k), 0)
    
    cv2.imwrite(OUT_RGB_PATH + name + '_blur' +'_.png', img)
    # imwrite(OUT_DEPTH_PATH + name + '_blur'+'_depth.tif', depth)
    cv2.imwrite(OUT_MASK_PATH +name + '_blur' +'_mask.png', mask)

def img_rotate(img, mask, name):
    rot = random.randint(10, 45)
    
    h, w = img.shape[:2]

    p_m = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
    m_m = cv2.getRotationMatrix2D((w/2, h/2), -rot, 1)

    rot_img_p = cv2.warpAffine(img, p_m, (w, h))
    # rot_depth_p = cv2.warpAffine(depth, p_m, (w, h))
    rot_mask_p = cv2.warpAffine(mask, p_m, (w, h))


    cv2.imwrite(OUT_RGB_PATH + name + '_rot_p' +'_.png', rot_img_p)
    # imwrite(OUT_DEPTH_PATH + name + '_rot_p'+'_depth.tif', rot_depth_p)
    cv2.imwrite(OUT_MASK_PATH +name + '_rot_p' +'_mask.png', rot_mask_p)

    rot_img_m = cv2.warpAffine(img, m_m, (w, h))
    # rot_depth_m = cv2.warpAffine(depth, m_m, (w, h))
    rot_mask_m = cv2.warpAffine(mask, m_m, (w, h))

    cv2.imwrite(OUT_RGB_PATH + name + '_rot_m' +'_.png', rot_img_m)
    # imwrite(OUT_DEPTH_PATH + name + '_rot_m'+'_depth.tif', rot_depth_m)
    cv2.imwrite(OUT_MASK_PATH +name + '_rot_m' +'_mask.png', rot_mask_m)

def cutmix(img, mask, bg, name):
    bg = cv2.resize(bg, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    rand_x = random.randint(0, 200)
    rand_y = random.randint(0, 200)

    shift_x = rand_x
    shift_y = rand_y
    h, w = img.shape[:2]

    # plus affine
    if random.randint(0, 1) == 1:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    else:
        M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    img = cv2.warpAffine(img, M, (w, h))
    # aff_depth_p = cv2.warpAffine(depth, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))
    
    # mask = mask[:, :, 0]
    binary_mask = np.where(mask >= 200, 1, 0).astype(np.uint8)
    
    img *= binary_mask


    bg_background = bg * np.where(binary_mask == 1, 0, 1).astype(np.uint8)
    bg_background += img

    cv2.imwrite(OUT_RGB_PATH + name + '_bg' +'_.png', bg_background)
    
    cv2.imwrite(OUT_MASK_PATH +name + '_bg' +'_mask.png', mask)
    

    
    

def save_imgs(rgb, mask, name):
    cv2.imwrite(OUT_RGB_PATH + name + '_original' +'_.png', rgb)
    # imwrite(OUT_DEPTH_PATH + name + '_original'+'_depth.tif', depth)
    cv2.imwrite(OUT_MASK_PATH +name + '_original' +'_mask.png', mask)
    print('save_original')


if __name__ == '__main__': 
    print(rgb_list[0], mask_list[0])
    i = 1

    bg_range = range(len(bg_list))
    for idx in range(len(rgb_list)):
        img = cv2.imread(rgb_list[idx])
        mask = cv2.imread(mask_list[idx])

        
        
        name = rgb_list[idx].split('/')[4] + '_' + str(i)
        
        save_imgs(img, mask, name)
        # img_shift(img, mask, name)
        img_blur(img, mask, name)
        img_rotate(img, mask, name)
        # for j in range(3):
        #     bg_idx = random.choice(bg_range)
        #     bg = cv2.imread(bg_list[bg_idx])
        #     cutmix(img, mask, bg, name)
        i+=1

        
        
        

        
        
