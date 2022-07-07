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
parser.add_argument("--obj_mask_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/obj_mask/')
parser.add_argument("--bg_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/bg/')
parser.add_argument("--output_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/augment/')

args = parser.parse_args()

"""
inputs (rgb, mask segObj, bg)
    rgb : RGB image
    mask : segmentation mask
    segObj : segmentation object mask
    bg : Background image
"""

class ImageAugmentationLoader():
    def __init__(self, args):
        self.RGB_PATH = args.rgb_path
        self.MASK_PATH = args.mask_path
        self.OBJ_MASK_PATH = args.obj_mask_path
        self.BG_PATH = args.bg_path
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'mask/'
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_MASK_PATH, exist_ok=True)

        self.rgb_list = glob.glob(os.path.join(self.RGB_PATH+'*.jpg'))
        self.rgb_list = natsort.natsorted(self.rgb_list,reverse=True)

        self.mask_list = glob.glob(os.path.join(self.MASK_PATH+'*.png'))
        self.mask_list = natsort.natsorted(self.mask_list,reverse=True)

        self.obj_mask_list = glob.glob(os.path.join(self.OBJ_MASK_PATH+'*.png'))
        self.obj_mask_list = natsort.natsorted(self.obj_mask_list,reverse=True)

        self.bg_list = glob.glob(os.path.join(self.BG_PATH +'*.jpg' ))

    
    def check_image_size(self):
        if len(self.rgb_list) != len(self.mask_list):
            raise Exception('RGB image와 MASK image수가 다릅니다!')


    def get_rgb_list(self):
        return self.rgb_list


    def get_mask_list(self):
        return self.mask_list

    def get_obj_mask_list(self):
        return self.obj_mask_list


    def get_bg_list(self):
        return self.bg_list

    
    def histogram_equalization(self, rgb):
        hist, _ = np.histogram(rgb.flatten(), 256,[0,256])

        cdf = hist.cumsum()

        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

        cdf = np.ma.filled(cdf_m,0).astype('uint8')

        return cdf[rgb]

    
    def bg_brightness(self, bg):
        random_val = np.random.uniform(low=0.5, high=1.0)
        val = 100
        val *= random_val
        array = np.full(bg.shape, (val, val, val), dtype=np.uint8)
        return cv2.add(bg, array)

    def bg_rotation(self, bg):
        rot = random.randint(5, 90)
        reverse = random.randint(1, 2)

        if reverse == 2:
            rot *= -1

        h, w = bg_img.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
        bg_img = cv2.warpAffine(bg_img, rot_mat, (w, h))


                

    def save_images(self, rgb, mask, prefix):
        print(prefix)
        cv2.imwrite(self.OUT_RGB_PATH + prefix +'_.png', rgb)
        cv2.imwrite(self.OUT_MASK_PATH + prefix + '_mask.png', mask)


    def change_image(self, img_idx, rgb, mask, obj_mask, options):
        """
        rgb = (H, W, 3)
        mask = (H, W, 3 , 124
        obj_mask = (H, W, 3)"""
        # TODO : Add random crop 

        crop_times = options['crop_times']

        for crop_idx in range(crop_times):
            scale = tf.random.uniform([], 0.5, 1)
            # rgb.shape -> h ,w, 3
            new_w = rgb.shape[1] * scale
            new_h = new_w * 1.6

            crop_img = rgb.copy()
            crop_mask = mask.copy()
            crop_mask = tf.expand_dims(crop_mask, axis=-1)
            crop_img = tf.cast(crop_img, tf.uint8)

            concat_img = tf.concat([crop_img, crop_mask], axis=-1)
            concat_img = tf.image.random_crop(concat_img, size=[new_h, new_w, 4])
            
            crop_img = concat_img[:, :, :3]
            crop_mask = concat_img[:, :, 3:]
            
            self.save_images(rgb=crop_img.numpy(), mask=crop_mask.numpy(), prefix='_{0}_original_{1}_crop_'.format(img_idx, crop_idx))     


        
        br_rgb = self.bg_brightness(bg=rgb.copy())     
        self.save_images(rgb=br_rgb.copy(), mask=mask.copy(), prefix='_{0}_brightness_'.format(img_idx))                                                    

        for crop_idx in range(2):
            scale = tf.random.uniform([], 0.5, 1)
            # rgb.shape -> h ,w, 3
            new_w = rgb.shape[1] * scale
            new_h = new_w * 1.6

            crop_img = br_rgb.copy()
            crop_mask = mask.copy()
            crop_mask = tf.expand_dims(crop_mask, axis=-1)
            crop_img = tf.cast(crop_img, tf.uint8)

            concat_img = tf.concat([crop_img, crop_mask], axis=-1)
            concat_img = tf.image.random_crop(concat_img, size=[new_h, new_w, 4])
            
            crop_img = concat_img[:, :, :3]
            crop_mask = concat_img[:, :, 3:]
            
            self.save_images(rgb=crop_img.numpy(), mask=crop_mask.numpy(), prefix='_{0}_brightness_{1}_crop_'.format(img_idx, crop_idx))  

        
        histogram_rgb = self.histogram_equalization(rgb=rgb.copy())     
        self.save_images(rgb=histogram_rgb.copy(), mask=mask.copy(), prefix='_{0}_histogram_'.format(img_idx))                                                    

        for crop_idx in range(2):
            scale = tf.random.uniform([], 0.5, 1)
            # rgb.shape -> h ,w, 3
            new_w = rgb.shape[1] * scale
            new_h = new_w * 1.6

            crop_img = histogram_rgb.copy()
            crop_mask = mask.copy()
            crop_mask = tf.expand_dims(crop_mask, axis=-1)
            crop_img = tf.cast(crop_img, tf.uint8)

            concat_img = tf.concat([crop_img, crop_mask], axis=-1)
            concat_img = tf.image.random_crop(concat_img, size=[new_h, new_w, 4])
            
            crop_img = concat_img[:, :, :3]
            crop_mask = concat_img[:, :, 3:]
            
            self.save_images(rgb=crop_img.numpy(), mask=crop_mask.numpy(), prefix='_{0}_histogram_{1}_crop_'.format(img_idx, crop_idx))  


        


        # # TODO change argmax

        # # binary_mask = binary_mask[:, :, 0]
        # contour_idx = 0

        # obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_RGB2GRAY)
        # obj_mask = obj_mask.astype(np.uint8)

        # obj_idx = np.unique(obj_mask)
        # obj_idx = np.delete(obj_idx, 0)

        # for idx in obj_idx: # 1 ~ obj nums
        #     binary_mask = np.where(obj_mask==idx, 255, 0)
            
        #     binary_mask = binary_mask.astype(np.uint8)
        #     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            

        #     for contour in contours:
        #         contour_idx += 1
        #         # 마스크의 contour를 이용하여 합성 할 영역의 bounding box를 계산
        #         x, y, w, h = cv2.boundingRect(contour)

        #         # 합성할 레퍼런스(background : bg) 이미지 랜덤으로 불러오기
        #         rnd_int = random.randint(0, len(self.bg_list)-1)

        #         bg_img = cv2.imread(self.bg_list[rnd_int]) # shape : (h, w, 3)
        #         bg_img = cv2.resize(bg_img, (w, h)) # bounding box 크기만큼 resizing


        #         if options['blur'] == True:
        #             k = random.randrange(3,21,2)
        #             bg_img = cv2.GaussianBlur(bg_img, (k,k), 0)
                

        #         if options['brightness'] == True:
        #             bg_img = self.bg_brightness(bg=bg_img)
                
        #         binary_mask_copy = binary_mask.copy()
        #         binary_mask_copy = np.expand_dims(binary_mask_copy, axis=-1)

        #         copy_mask = np.where(binary_mask_copy[y:y+h, x:x+w] == 255, bg_img, rgb[y:y+h, x:x+w])
                

        #         rgb[y:y+h, x:x+w] = copy_mask
        #         self.save_images(rgb=rgb, mask=mask, prefix='_{0}_obj_{1}_contour_{2}_original'.format(img_idx, idx, contour_idx))

        #         if options['histogram_eq'] == True:
        #             his_eq_rgb = self.histogram_equalization(rgb=rgb.copy())
        #             self.save_images(rgb=his_eq_rgb, mask=mask, prefix='_{0}_obj_{1}_contour_{2}_histogram_eq'.format(img_idx, idx, contour_idx))
                
        #         # random crop
        #         aug_crop_times = options['aug_crop_times']

        #         for crop_idx in range(aug_crop_times):
        #             scale = tf.random.uniform([], 0.5, 1.4)
        #         # rgb.shape -> h ,w, 3
        #             new_w = (rgb.shape[1] * 0.5) * scale
        #             new_h = new_w * 1.6

        #             crop_img = rgb.copy()
        #             crop_mask = mask.copy()
        #             crop_mask = tf.expand_dims(crop_mask, axis=-1)
        #             crop_img = tf.cast(crop_img, tf.uint8)

        #             concat_img = tf.concat([crop_img, crop_mask], axis=-1)
        #             concat_img = tf.image.random_crop(concat_img, size=[new_h, new_w, 4])
                    
        #             crop_img = concat_img[:, :, :3]
        #             crop_mask = concat_img[:, :, 3:]
        #             self.save_images(rgb=crop_img.numpy(), mask=crop_mask.numpy(), prefix='_{0}_obj_{1}_contour_{2}_crop_{3}_'.format(img_idx, idx, contour_idx, crop_idx))                                                            





if __name__ == '__main__':
    """
    mask image pixel value =  124.0
    """
    image_loader = ImageAugmentationLoader(args=args)
    rgb_list = image_loader.get_rgb_list()
    mask_list = image_loader.get_mask_list()
    obj_mask_list = image_loader.get_obj_mask_list()
    bg_list = image_loader.get_bg_list()

    change_img_options = {
            'blur': True,
            'brightness': True,
            'histogram_eq': True,
            'crop_times': 5,
            'aug_crop_times': 2,
             }

    for idx in range(len(rgb_list)):
        original_rgb = cv2.imread(rgb_list[idx])

        original_mask = cv2.imread(mask_list[idx])
        original_mask = original_mask[:, :, 0]

        original_obj_mask = cv2.imread(obj_mask_list[idx])

        rgb = original_rgb.copy()
        mask = original_mask.copy()
        obj_mask = original_obj_mask.copy()


        # rgb = cv2.resize(rgb, [1080, 1920], interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, [1080, 1920], interpolation=cv2.INTER_NEAREST)
        # obj_mask = cv2.resize(obj_mask, [1080, 1920], interpolation=cv2.INTER_NEAREST)
        
    
        # save original
        image_loader.save_images(rgb=original_rgb, mask=original_mask, prefix='_{0}_original'.format(idx))

        # save change only image
        # image_loader.change_image(img_idx=idx, rgb=rgb, mask=mask, obj_mask=obj_mask, options=None)
        # image_loader.save_images(rgb=change_rgb, mask=change_mask, prefix='_{0}_change'.format(idx))

        # save change with random blur & rotation
        
        image_loader.change_image(img_idx=idx, rgb=rgb, mask=mask, obj_mask=obj_mask, options=change_img_options)
            # image_loader.save_images(rgb=change_rgb, mask=change_mask, prefix='_{0}_change_{1}'.format(idx, change_times))

        

        

