import numpy as np
import cv2
from imageio import imread, imwrite
import glob
import os
import argparse
from utils import load_imgs, canny_edge, find_contours

parser = argparse.ArgumentParser()
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_labeling/data/test/mask_result/')
args = parser.parse_args()
IMAGE_PATH = args.mask_path



img_list = glob.glob(os.path.join(IMAGE_PATH,'*.png'))
img_list.sort()


if __name__ == '__main__': 
    for idx in range(len(img_list)):
        img = imread(img_list[idx])
        mask, img = load_imgs(img)
        mask = canny_edge(mask, False, False)
        draw_mask = find_contours(mask, img, (0, 255, 0))