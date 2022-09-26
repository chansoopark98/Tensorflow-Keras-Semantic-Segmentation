from typing import Union
import numpy as np
import cv2
import glob
import os
import argparse
import natsort
import tensorflow as tf
import random
import albumentations as A

name = 'human_fashion_1_dataset'

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/choose/{0}/select/rgb/'.format(name))
parser.add_argument("--mask_path",     type=str,   help="raw mask path", default='./data_augmentation/raw_data/choose/{0}/select/gt/'.format(name))
parser.add_argument("--bg_path",     type=str,   help="bg image path, Convert raw rgb image using mask area", default='./data_augmentation/raw_data/bg_img/save_bg/rgb/')
parser.add_argument("--output_path",     type=str,   help="Path to save the conversion result", default='./data_augmentation/raw_data/choose/{0}/compose/'.format(name))

args = parser.parse_args()

class ImageAugmentationLoader():
    def __init__(self, args):
        """
        Args
            args  (argparse) : inputs (rgb, mask segObj, bg)
                >>>    rgb : RGB image.
                >>>    mask : segmentation mask.
                >>>    segObj : segmentation object mask.
                >>>    label_map : segmentation mask(label) information.
                >>>    bg : Background image.
        """
        self.RGB_PATH = args.rgb_path
        self.MASK_PATH = args.mask_path
        self.BG_PATH = args.bg_path
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'gt/'
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_MASK_PATH, exist_ok=True)

        self.rgb_list = glob.glob(os.path.join(self.RGB_PATH+'*.jpg'))
        self.rgb_list = natsort.natsorted(self.rgb_list,reverse=True)

        self.mask_list = glob.glob(os.path.join(self.MASK_PATH+'*.png'))
        self.mask_list = natsort.natsorted(self.mask_list,reverse=True)

        self.bg_list = glob.glob(os.path.join(self.BG_PATH +'*.jpg' ))
        
        # Check your data (RGB file samples = Mask file samples)
        self.check_image_len() 


    def check_image_len(self):
        """
            Check rgb, mask, obj mask sample counts
        """
        rgb_len = len(self.rgb_list)
        mask_len = len(self.mask_list)

        if rgb_len != mask_len:
            raise Exception('RGB Image files : {0}, Mask Image files : {1}. Check your image and mask files '
                            .format(rgb_len, mask_len))


    def get_rgb_list(self) -> list:
        """
            return rgb list instance
        """
        return self.rgb_list

    def get_mask_list(self) -> list:
        """
            return mask list instance
        """
        return self.mask_list

    def get_bg_list(self) -> list:
        """
            return bg image list instance
        """
        return self.bg_list
    
    def resize_bg_image(self, bg_image: np.ndarray, rgb_shape: tuple):
        h, w = rgb_shape[:2]
        bg_image = cv2.resize(bg_image, (w, h))

        return bg_image
        
    def image_random_translation(self, rgb: np.ndarray, mask: np.ndarray,
                                 min_dx: int, min_dy: int,
                                 max_dx: int, max_dy: int) -> Union[np.ndarray, np.ndarray]:
        """
            Random translation function   
            Args:
                rgb        (np.ndarray) : (H,W,3) Image.
                mask       (np.ndarray) : (H,W,1) Image.
                min_dx  (int)      : Minimum value of pixel movement distance based on the x-axis when translating an image.
                min_dy  (int)      : Minimum value of pixel movement distance based on the y-axis when translating an image.
                max_dx  (int)      : Maximum value of pixel movement distance based on the x-axis when translating an image.
                max_dy  (int)      : Maximum value of pixel movement distance based on the y-axis when translating an image.
                
        """
        random_dx = random.randint(min_dx, max_dx)
        random_dy = random.randint(min_dy, max_dy)

        if tf.random.uniform([]) > 0.5:
            random_dx *= -1

        if tf.random.uniform([]) > 0.5:
            random_dy *= -1

        rows, cols = rgb.shape[:2]
        trans_mat = np.float64([[1, 0, random_dx], [0, 1, random_dy]])

        trans_rgb = cv2.warpAffine(rgb, trans_mat, (cols, rows))
        trans_mask = cv2.warpAffine(mask, trans_mat, (cols, rows))

        return trans_rgb, trans_mask

    def save_images(self, rgb: np.ndarray, mask: np.ndarray, prefix: str):
        """
            Save image and mask
            Args:
                rgb     (np.ndarray) : (H,W,3) Image.
                mask    (np.ndarray) : (H,W,1) Image.
                prefix  (str)        : The name of the image to be saved.
        """
        cv2.imwrite(self.OUT_RGB_PATH + prefix +'_.jpg', rgb)
        cv2.imwrite(self.OUT_MASK_PATH + prefix + '_mask.png', mask)

                                                              
if __name__ == '__main__':
    """
    Image augmentation can be selected according to the option using the internal function of ImageAugmentationLoader.
    """
    from tqdm import tqdm

    image_loader = ImageAugmentationLoader(args=args)
    rgb_list = image_loader.get_rgb_list()
    mask_list = image_loader.get_mask_list()
    bg_list = image_loader.get_bg_list()
    
    # scale_rotate = A.ShiftScaleRotate(rotate_limit=40, scale_limit=0.1, )
    random_sun = A.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=4, src_radius=50, )
    random_shadow = A.RandomShadow()
    random_snow = A.RandomSnow(brightness_coeff=1.2)
    img_aug = A.Compose([random_sun, random_shadow, random_snow])

    # for idx in range(len(rgb_list)):
    for idx in tqdm(range(len(rgb_list)), total=len(rgb_list)):
        original_rgb = cv2.imread(rgb_list[idx])
        
        original_mask = cv2.imread(mask_list[idx])
        original_mask = np.where(original_mask>=1, 255, 0).astype(np.uint8)
        
        # image aug augmentation
        transformed = img_aug(image=original_rgb.copy(), mask=original_mask.copy())
        aug_rgb = transformed['image']
        aug_mask = transformed['mask']
        image_loader.save_images(rgb=aug_rgb, mask=aug_mask, prefix='{0}_idx_{1}_rgb_aug_'.format(name, idx))

        # change bg image
        sift_rgb, sift_mask = image_loader.image_random_translation(rgb=original_rgb.copy(), mask=original_mask.copy(),min_dx=15, min_dy=10, max_dx=150, max_dy=100)

        bg_rnd_idx = random.randint(0, len(bg_list)-1)
        original_bg = cv2.imread(bg_list[bg_rnd_idx])
        
        bg_image = image_loader.resize_bg_image(bg_image=original_bg, rgb_shape=sift_rgb.shape)
                
        change_bg = np.where(
                    sift_mask == 255, sift_rgb, bg_image)
        image_loader.save_images(rgb=change_bg.copy(), mask=sift_mask.copy(), prefix='{0}_idx_{1}_change_bg_'.format(name, idx))


        # 1. Default image (1) + Random rotation (1) + Random translation (1)
        # image_loader.save_images(rgb=rgb.copy(), mask=mask.copy(), prefix='idx_{0}_original_'.format(idx))

