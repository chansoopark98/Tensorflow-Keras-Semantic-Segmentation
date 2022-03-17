import numpy as np
import cv2
from imageio import imread
import glob
import os
import argparse
from utils import load_imgs, canny_edge, find_contours

parser = argparse.ArgumentParser()
parser.add_argument("--image_path",     type=str,   help="raw image path", default='./data_labeling/data/img/')
args = parser.parse_args()
IMAGE_PATH = args.image_path

img_list = glob.glob(os.path.join(IMAGE_PATH,'*.jpg'))
img_list.sort()


if __name__ == '__main__': 
    for idx in range(len(img_list)):
        img = imread(img_list[idx])
        mask, img = load_imgs(img)
        mask = canny_edge(mask)
        draw_mask, draw_hole = find_contours(mask, img)

        cv2.imwrite('./data_labeling/data/gt/'+str(idx)+'_mask.png', draw_mask)
        cv2.imwrite('./data_labeling/data/gt/'+str(idx)+'_hole.png', draw_hole)