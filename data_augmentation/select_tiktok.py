import json
import cv2
import glob
import natsort
import os
import numpy as np
import PIL
from tqdm import tqdm

RAW_PATH = './data_augmentation/raw_data/TikTok_dataset/'
SAVE_RGB_PATH = RAW_PATH + 'rgb/'
SAVE_MASK_PATH = RAW_PATH + 'mask/'
SELECT_STEP = 5

os.makedirs(SAVE_RGB_PATH, exist_ok=True)
os.makedirs(SAVE_MASK_PATH, exist_ok=True)

raw_list = glob.glob(RAW_PATH + 'raw/*')
raw_list = natsort.natsorted(raw_list,reverse=True)

sample_dir_idx = 0
image_idx = 0

for samples in tqdm(raw_list, total=len(raw_list)):

    sample_dir_idx += 1

    rgb_list = glob.glob(os.path.join(samples+'/images/*.png'))
    rgb_list = natsort.natsorted(rgb_list, reverse=True)

    mask_list = glob.glob(os.path.join(samples+'/masks/*.png'))
    mask_list = natsort.natsorted(mask_list, reverse=True)

    for idx in range(0, len(rgb_list), SELECT_STEP):
        image = cv2.imread(rgb_list[idx])
        mask = cv2.imread(mask_list[idx])

        cv2.imwrite(SAVE_RGB_PATH + 'human_segmentation_dataset_3_tiktok_{0}_idx_{1}.jpg'.format(sample_dir_idx, idx), image)
        cv2.imwrite(SAVE_MASK_PATH + 'human_segmentation_dataset_3_tiktok_{0}_idx_{1}.png'.format(sample_dir_idx, idx), mask)


    