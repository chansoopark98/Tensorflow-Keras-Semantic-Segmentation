import json
import cv2
import glob
import natsort
import os
import numpy as np
import PIL
from tqdm import tqdm

RAW_PATH = './data_augmentation/raw_data/LASIESTA_human_dataset/rgb/'
SAVE_PATH = './data_augmentation/raw_data/LASIESTA_human_dataset/select/'
SAVE_RGB_PATH = SAVE_PATH + 'rgb/'
SAVE_MASK_PATH = SAVE_PATH + 'gt/'
SELECT_STEP = 10

os.makedirs(SAVE_RGB_PATH, exist_ok=True)
os.makedirs(SAVE_MASK_PATH, exist_ok=True)

raw_list = glob.glob(RAW_PATH + '/*')
raw_list = natsort.natsorted(raw_list,reverse=True)

sample_dir_idx = 0
image_idx = 0

for samples in tqdm(raw_list, total=len(raw_list)):

    print(samples)
    sample_dir_idx += 1

    
    rgb_dir_name = samples
    mask_dir_name = rgb_dir_name.replace('rgb', 'gt') + '-GT'

    rgb_list = glob.glob(os.path.join(rgb_dir_name+'/*.jpg'))
    rgb_list = natsort.natsorted(rgb_list, reverse=True)

    mask_list = glob.glob(os.path.join(mask_dir_name+'/*.png'))
    mask_list = natsort.natsorted(mask_list, reverse=True)
    

    for idx in range(0, len(rgb_list), SELECT_STEP):
        image = cv2.imread(rgb_list[idx])
        mask = cv2.imread(mask_list[idx])

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = np.where(mask>=1, 255, 0)
        
        mask = np.expand_dims(mask, axis=-1)
        # concat_mask = np.concatenate([mask, mask, mask], axis=-1).astype(np.uint8)
        # masked_image = image * (concat_mask / 255)
        # masked_image = masked_image.astype(np.uint8)
        # print(image.shape)
        # print(concat_mask.shape)
        # print(masked_image.shape)


        # vis_img = cv2.hconcat([image, concat_mask, masked_image])

        # cv2.imshow('test', vis_img)
        # cv2.waitKey(0)

        cv2.imwrite(SAVE_RGB_PATH + 'LASIESTA_human_dataset_{0}_idx_{1}.jpg'.format(sample_dir_idx, idx), image)
        cv2.imwrite(SAVE_MASK_PATH + 'LASIESTA_human_dataset_{0}_idx_{1}.png'.format(sample_dir_idx, idx), mask)


    