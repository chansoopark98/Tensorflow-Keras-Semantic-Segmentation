import numpy as np
import cv2
from imageio import imread, imwrite
import glob
import os
import argparse
from utils import load_imgs, canny_edge, find_contours

parser = argparse.ArgumentParser()
parser.add_argument("--image_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040533_대각_24cm_white_b1/rgb/')
parser.add_argument("--result_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040533_대각_24cm_white_b1/result/')


args = parser.parse_args()
IMAGE_PATH = args.image_path
RESULT_PATH = args.result_path
RESOLUTION = (640,480)
RGB_PATH = RESULT_PATH+'rgb/'
MASK_PATH = RESULT_PATH+'mask/'

os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(RGB_PATH, exist_ok=True)
os.makedirs(MASK_PATH, exist_ok=True)

img_list = glob.glob(os.path.join(IMAGE_PATH,'*.png'))
img_list.sort()



if __name__ == '__main__': 
    for idx in range(len(img_list)):
        img = cv2.imread(img_list[idx])


        date_name = img_list[idx].split('/')[4]
        print('data_name', date_name)
        
        # img_name = img_list[idx].split('/')[5].split('\\')[1].split('.')[0]
        img_name = img_list[idx].split('/')[4]
        print('img_name', img_name)
        mask, img = load_imgs(img, resize=RESOLUTION)
        mask = canny_edge(mask)
        draw_mask = find_contours(mask, img)

        if len(draw_mask) != 0:
            print('save_image')
            cv2.imwrite(RGB_PATH+date_name + '_' +img_name+'.png', img)
            cv2.imwrite(MASK_PATH+date_name + '_' +img_name+'_mask.png', draw_mask)
            