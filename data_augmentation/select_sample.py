from typing import Union
import numpy as np
import cv2
import glob
import os
import argparse
import natsort
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import math
import random

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/raw_datasets/human_segmentation_dataset_2/rgb/')
parser.add_argument("--mask_path",     type=str,   help="raw mask path", default='./data_augmentation/raw_data/raw_datasets/human_segmentation_dataset_2/mask/')
parser.add_argument("--output_path",     type=str,   help="Path to save the conversion result", default='./data_augmentation/raw_data/raw_datasets/human_segmentation_dataset_2/select/')

args = parser.parse_args()

# 0920 15:40 829까지 작업

class ImageAugmentationLoader():
    def __init__(self, args):
        """
        Args
            args  (argparse) : inputs (rgb, mask)
                >>>    rgb : RGB image.
                >>>    mask : segmentation mask.
        """
        self.RGB_PATH = args.rgb_path
        self.MASK_PATH = args.mask_path
        
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'mask/'
        
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_MASK_PATH, exist_ok=True)
        

        self.rgb_list = glob.glob(os.path.join(self.RGB_PATH+'*.jpg'))
        self.rgb_list = natsort.natsorted(self.rgb_list,reverse=True)

        self.mask_list = glob.glob(os.path.join(self.MASK_PATH+'*.png'))
        self.mask_list = natsort.natsorted(self.mask_list,reverse=True)


    def image_resize(self, rgb: np.ndarray, mask: np.ndarray, size=(1600, 900)) -> Union[np.ndarray, np.ndarray]:
        """
            Image resizing function    
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                mask     (np.ndarray) : (H,W,1) Image.
                size     (tuple)      : Image size to be adjusted.
        """
        resized_rgb = tf.image.resize(images=rgb, size=size, method=tf.image.ResizeMethod.BILINEAR)
        resized_rgb = resized_rgb.numpy().astype(np.uint8)

        resized_mask = tf.image.resize(images=mask, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_mask = resized_mask.numpy().astype(np.uint8)


        return resized_rgb, resized_mask

    def save_images(self, rgb, mask, prefix):
        
        cv2.imwrite(self.OUT_RGB_PATH + prefix +'_rgb.jpg', rgb)
        cv2.imwrite(self.OUT_MASK_PATH + prefix + '_mask.png', mask)

                                                              
if __name__ == '__main__':
    """
    Image augmentation can be selected according to the option using the internal function of ImageAugmentationLoader.
    """
    from tqdm import tqdm

    image_loader = ImageAugmentationLoader(args=args)
    rgb_list = image_loader.rgb_list
    mask_list = image_loader.mask_list

    # for idx in range(len(rgb_list)):
    for idx in tqdm(range(len(rgb_list)), total=len(rgb_list)):
        rgb_name = rgb_list[idx]
        
        original_rgb = cv2.imread(rgb_list[idx])
        original_mask = cv2.imread(mask_list[idx])

        original_rgb_shape = original_rgb.shape[:2]
        original_mask_shape = original_mask.shape[:2]

        if original_rgb_shape != original_mask_shape:
            
            
            h, w = original_rgb_shape
            original_mask = cv2.resize(original_mask, (w, h), interpolation=cv2.INTER_NEAREST)


        original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)
        original_mask = np.where(original_mask >= 1, 1, 0).astype(np.uint8)

    
        contours, _ = cv2.findContours(
                original_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        draw_contours = []

        rgb_shape = original_rgb.shape[:2]
        hw_area = rgb_shape[0] * rgb_shape[1]

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            print(hw_area, hw_area * 0.003,  area)
            if area <= (hw_area * 0.003):
                draw_contours.append(contours[i])

        zero_maks = np.zeros(original_mask.shape, np.uint8)
        zero_maks = cv2.drawContours(zero_maks, draw_contours, -1, 1, thickness=-1)

        original_mask += zero_maks
        original_mask = np.where(original_mask>=1, 255, 0).astype(np.uint8)
        original_mask = np.expand_dims(original_mask, axis=-1)

        concat_mask = np.concatenate([original_mask, original_mask, original_mask], axis=-1)

        concat_img = cv2.hconcat([original_rgb, concat_mask])

        winname = "test"
        cv2.namedWindow(winname)   # create a named window
        cv2.moveWindow(winname, 850, 500)   # Move it to (40, 30)

        cv2.imshow(winname, concat_img)
        key = cv2.waitKey(0)
        
        print(key)
        """
        q : 113 -> 그냥 저장
        1 : 49 -> roi 선택 후 저장
        """
        print(key)
        if key == 113:
            print('바로 저장')
            image_loader.save_images(rgb=original_rgb, mask=original_mask, prefix='human_segmentation_dataset_1_{0}_'.format(idx))
        
        elif key == 49:
            try:
                x, y, w, h = cv2.selectROI(winname, concat_img)
            except:
                cv2.destroyAllWindows()
                continue
            cv2.destroyAllWindows()
            cropped_rgb = original_rgb[y:y+h, x:x+w]
            cropped_mask = original_mask[y:y+h, x:x+w]
            
            image_loader.save_images(rgb=cropped_rgb, mask=cropped_mask, prefix='human_segmentation_dataset_1_{0}_crop_'.format(idx))

        else:
            print('key 입력 오류')
            continue


        