import json
import cv2
import glob
import natsort
import os
import numpy as np
import PIL
from tqdm import tqdm
import matplotlib.pyplot as plt


RAW_PATH = './data_augmentation/raw_data/LV-MHP-v2/train/rgb/'
SAVE_PATH = './data_augmentation/raw_data/LV-MHP-v2/train/select/'
SAVE_RGB_PATH = SAVE_PATH + 'rgb/'
SAVE_MASK_PATH = SAVE_PATH + 'gt/'
SELECT_STEP = 10

os.makedirs(SAVE_RGB_PATH, exist_ok=True)
os.makedirs(SAVE_MASK_PATH, exist_ok=True)

raw_list = glob.glob(RAW_PATH)
raw_list = natsort.natsorted(raw_list,reverse=True)

sample_dir_idx = 0
image_idx = 0

for samples in tqdm(raw_list, total=len(raw_list)):

    
    sample_dir_idx += 1

    
    rgb_dir_name = samples
    mask_dir_name = rgb_dir_name.replace('rgb', 'gt')

    rgb_list = glob.glob(os.path.join(rgb_dir_name+'/*.jpg'))
    rgb_list = natsort.natsorted(rgb_list, reverse=True)

    # mask_list = glob.glob(os.path.join(mask_dir_name+'/*.png'))
    # mask_list = natsort.natsorted(mask_list, reverse=True)
    

    for idx in range(len(rgb_list)):
        print(rgb_list[idx]) # ./data_augmentation/raw_data/LV-MHP-v2/train/rgb/25790.jpg
        
        rgb_sample_name = rgb_list[idx].split('/')[6]
        idx_name = rgb_sample_name.split('.')[0]
        
        rgb_name = rgb_list[idx]
        mask_idx = RAW_PATH.replace('rgb', 'gt')

        mask_files = glob.glob(os.path.join(mask_idx+ idx_name +'*.png'))

        
        image = cv2.imread(rgb_list[idx])

        mask = np.zeros(image.shape[:2])

        for mask_file in mask_files:
            sample_mask = cv2.imread(mask_file)
            
            # sample_mask = cv2.cvtColor(sample_mask, cv2.COLOR_BGR2GRAY)
            sample_mask = sample_mask[:, :, 2]
            plt.imshow(sample_mask)
            plt.show()
            sample_mask = np.where(sample_mask>=1, 1, 0)    
            mask += sample_mask

        mask = np.where(mask>=1, 1, 0).astype(np.uint8)
        
        

        contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        zero_maks = np.zeros(mask.shape, np.uint8)
        zero_maks = cv2.drawContours(zero_maks, contours, -1, 1, thickness=-1)


        mask += zero_maks
        mask = np.where(mask>=1, 255, 0).astype(np.uint8)


        draw_contours = []

        rgb_shape = image.shape[:2]
        
        hw_area = rgb_shape[0] * rgb_shape[1]
        
        blank_contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(blank_contours)):
            area = cv2.contourArea(blank_contours[i])
            
            if area <= (hw_area * 0.003):
                draw_contours.append(blank_contours[i])

        black_mask = cv2.drawContours(mask, draw_contours, -1, 0, thickness=-1)

        mask = np.where(black_mask==0, 0, mask)


        mask = np.expand_dims(mask, axis=-1)

        concat_mask = np.concatenate([mask, mask, mask], axis=-1).astype(np.uint8)
        masked_image = image * (concat_mask / 255)
        masked_image = masked_image.astype(np.uint8)
        print(image.shape)
        print(concat_mask.shape)
        print(masked_image.shape)


        vis_img = cv2.hconcat([image, concat_mask, masked_image])

        cv2.imshow('test', vis_img)
        cv2.waitKey(0)

        # cv2.imwrite(SAVE_RGB_PATH + 'LV-MHP-v2_train_human_dataset_{0}_idx_{1}.jpg'.format(sample_dir_idx, idx), image)
        # cv2.imwrite(SAVE_MASK_PATH + 'LV-MHP-v2_train_human_dataset_{0}_idx_{1}.png'.format(sample_dir_idx, idx), mask)


    