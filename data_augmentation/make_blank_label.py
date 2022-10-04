import numpy as np
import cv2
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/bg_img/select/total/')
# parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/bg_img/select/select_rgb/')
# parser.add_argument("--obj_mask_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/obj_mask/')
# parser.add_argument("--bg_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/bg_img/select/select_rgb/')
parser.add_argument("--output_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/bg_img/save_bg/')

args = parser.parse_args()


if __name__ == '__main__':
    
    os.makedirs(args.output_path, exist_ok=True)

    rgb_path = os.path.join(args.rgb_path, '*.jpg')
    rgb_list = glob.glob(rgb_path)

    idx = 0
    for rgb_idx in rgb_list:
        print(rgb_idx)
        idx += 1
        file_name = rgb_idx.split('/')[5]
        file_name = file_name.split('.')[0]
        
        # if not os.path.isfile(args.obj_mask_path + file_name + '.png'):
        rgb_img = cv2.imread(rgb_idx)
        zero_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
        zero_mask = np.expand_dims(zero_mask, axis=-1)

        cv2.imwrite(args.output_path + 'rgb/' + 'bg2_image_idx_{0}_'.format(idx) + '_rgb.jpg', rgb_img)
        cv2.imwrite(args.output_path + 'gt/' + ' bg2_image_idx_{0}_'.format(idx) + '_mask.png', zero_mask)
        # cv2.imwrite(args.obj_mask_path + file_name + '.png', zero_mask)