import numpy as np
import cv2
from imageio import imread, imwrite
import glob
import os
import argparse
import natsort
import tensorflow as tf
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/rgb/')
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/mask/')
parser.add_argument("--bg_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/bg/')
parser.add_argument("--output_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/augment/')

args = parser.parse_args()


class ImageAugmentationLoader():
    def __init__(self, args):
        self.RGB_PATH = args.rgb_path
        self.MASK_PATH = args.mask_path
        self.BG_PATH = args.bg_path
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'mask/'

        self.rgb_list = glob.glob(os.path.join(self.RGB_PATH+'*.jpg'))
        self.rgb_list = natsort.natsorted(self.rgb_list,reverse=True)

        self.mask_list = glob.glob(os.path.join(self.MASK_PATH+'*.png'))
        self.mask_list = natsort.natsorted(self.mask_list,reverse=True)

        self.bg_list = glob.glob(os.path.join(self.BG_PATH +'*.jpg' ))

    
    def check_image_size(self):
        if len(self.rgb_list) != len(self.mask_list):
            raise Exception('RGB image와 MASK image수가 다릅니다!')


    def get_rgb_list(self):
        return self.rgb_list


    def get_mask_list(self):
        return self.mask_list


    def get_bg_list(self):
        return self.bg_list


if __name__ == '__main__':
    """
    mask image pixel value =  124.0
    """
    image_loader = ImageAugmentationLoader(args=args)
    rgb_list = image_loader.get_rgb_list()
    mask_list = image_loader.get_mask_list()
    bg_list = image_loader.get_bg_list()


    for idx in range(len(rgb_list)):
        rgb = cv2.imread(rgb_list[idx])
        mask = cv2.imread(mask_list[idx])

        mask = mask[:, :, 0]
        binary_mask = np.where(mask == 124, 1, 0).astype(np.uint8)
        binary_mask = np.expand_dims(binary_mask, axis=-1)
        # mask를 이용하여 contour를 계산
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 계산된 contour 개수만큼 반복 연산
        print('Contours shape : ', len(contours))
        for contour in contours:
            
            # 마스크의 contour를 이용하여 합성 할 영역의 bounding box를 계산
            x, y, w, h = cv2.boundingRect(contour)

            # 합성할 레퍼런스(background : bg) 이미지 랜덤으로 불러오기
            rnd_int = random.randint(0, len(bg_list)-1)

            bg_img = cv2.imread(bg_list[rnd_int]) # shape : (h, w, 3)
            bg_img = cv2.resize(bg_img, (w, h)) # bounding box 크기만큼 resizing

            
            copy_mask = np.where(binary_mask[y:y+h, x:x+w] == 1, bg_img, rgb[y:y+h, x:x+w])

            rgb[y:y+h, x:x+w] = copy_mask

            plt.imshow(rgb)
            plt.show()

            
        
        

        

