import numpy as np
import cv2
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/rgb/')
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/mask/')
parser.add_argument("--obj_mask_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/obj_mask/')
parser.add_argument("--bg_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/bg/')
parser.add_argument("--output_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/augment/')

args = parser.parse_args()


if __name__ == '__main__':

    rgb_path = os.path.join(args.rgb_path, '*.jpg')
    rgb_list = glob.glob(rgb_path)

    for rgb_idx in rgb_list:
        file_name = rgb_idx.split('/')[4]
        file_name = file_name.split('.')[0]
        
        if not os.path.isfile(args.obj_mask_path + file_name + '.png'):
            rgb_img = cv2.imread(rgb_idx)
            zero_mask = np.zeros(rgb_img.shape, dtype=np.uint8)
            cv2.imwrite(args.mask_path + file_name + '.png', zero_mask)
            cv2.imwrite(args.obj_mask_path + file_name + '.png', zero_mask)