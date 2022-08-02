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
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/rgb/')
parser.add_argument("--mask_path",     type=str,   help="raw mask path", default='./data_augmentation/raw_data/mask/')
parser.add_argument("--obj_mask_path",     type=str,   help="raw obj mask path", default='./data_augmentation/raw_data/obj_mask/')
parser.add_argument("--label_map_path",     type=str,   help="CVAT's labelmap.txt path", default='./data_augmentation/raw_data/labelmap.txt')
parser.add_argument("--bg_path",     type=str,   help="bg image path, Convert raw rgb image using mask area", default='./data_augmentation/raw_data/bg/')
parser.add_argument("--output_path",     type=str,   help="Path to save the conversion result", default='./data_augmentation/raw_data/augment/')

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
        self.OBJ_MASK_PATH = args.obj_mask_path
        self.LABEL_MAP_PATH = args.label_map_path
        self.BG_PATH = args.bg_path
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'mask/'
        self.OUT_VIS_MASK_PATH = self.OUTPUT_PATH + 'visual_mask/'
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_MASK_PATH, exist_ok=True)
        os.makedirs(self.OUT_VIS_MASK_PATH, exist_ok=True)

        self.rgb_list = glob.glob(os.path.join(self.RGB_PATH+'*.jpg'))
        self.rgb_list = natsort.natsorted(self.rgb_list,reverse=True)

        self.mask_list = glob.glob(os.path.join(self.MASK_PATH+'*.png'))
        self.mask_list = natsort.natsorted(self.mask_list,reverse=True)

        self.obj_mask_list = glob.glob(os.path.join(self.OBJ_MASK_PATH+'*.png'))
        self.obj_mask_list = natsort.natsorted(self.obj_mask_list,reverse=True)

        self.bg_list = glob.glob(os.path.join(self.BG_PATH +'*.jpg' ))
        
        # Check your data (RGB file samples = Mask file samples)
        self.check_image_len() 

        # Get label information from labelmap.txt (Segmentation mask 1.1 format)
        self.label_list = self.get_label_list()


    def get_label_list(self) -> list:
        """
            Get label information from labelmap.txt
        """
        label_list = []

        with open(self.LABEL_MAP_PATH, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx != 0:
                    split_line = line.split(':')

                    class_name = split_line[0]
                    string_rgb = split_line[1]

                    r, g, b = string_rgb.split(',')
                    r = int(r)
                    g = int(g)
                    b = int(b)

                    output = {'class_name': class_name,
                              'class_idx': idx-1,
                              'rgb': (r, g, b)}

                    label_list.append(output)

        return label_list


    def check_image_len(self):
        """
            Check rgb, mask, obj mask sample counts
        """
        rgb_len = len(self.rgb_list)
        mask_len = len(self.mask_list)
        obj_mask_len = len(self.obj_mask_list)

        if rgb_len != mask_len:
            raise Exception('RGB Image files : {0}, Mask Image files : {1}. Check your image and mask files '
                            .format(rgb_len, mask_len))
        
        if rgb_len != obj_mask_len:
            raise Exception('RGB Image files : {0}, Object Mask Image files : {1}. Check your image and obj mask files '
                            .format(rgb_len, obj_mask_len))


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


    def get_obj_mask_list(self) -> list:
        """
            return obj mask list instance
        """
        return self.obj_mask_list


    def get_bg_list(self) -> list:
        """
            return bg image list instance
        """
        return self.bg_list


    def image_resize(self, rgb: np.ndarray, mask: np.ndarray,
                     obj_mask: np.ndarray = None, size=(1600, 900)) -> Union[np.ndarray, np.ndarray]:
        """
            Image resizing function    
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                mask     (np.ndarray) : (H,W,1) Image.
                obj_mask (np.ndarray) : (H,W,1) Image, default is None.
                size     (tuple)      : Image size to be adjusted.
        """
        resized_rgb = tf.image.resize(images=rgb, size=size, method=tf.image.ResizeMethod.BILINEAR)
        resized_rgb = resized_rgb.numpy().astype(np.uint8)

        resized_mask = tf.image.resize(images=mask, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_mask = resized_mask.numpy().astype(np.uint8)

        if obj_mask is not None:
            resized_obj_mask = tf.image.resize(images=obj_mask, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            resized_obj_mask = resized_obj_mask.numpy().astype(np.uint8)

            return resized_rgb, resized_mask, resized_obj_mask
        
        else:
            return resized_rgb, resized_mask
        
    
    def image_histogram_equalization(self, rgb: np.ndarray) -> np.ndarray:
        """
            Image histogram equalization function
            Args:
                rgb (np.ndarray) : (H,W,3) Image.    
        """
        split_r, split_g, split_b = cv2.split(rgb)

        hist_r = cv2.equalizeHist(split_r)
        hist_g = cv2.equalizeHist(split_g)
        hist_b = cv2.equalizeHist(split_b)

        equ = cv2.merge((hist_r, hist_g, hist_b))

        return equ

    def image_random_bluring(self, rgb: np.ndarray, gaussian_min: int = 7,
                             gaussian_max: int = 21) -> np.ndarray:
        """
            NxN random gaussianBlurring function   
            Args:
                rgb           (np.ndarray) : (H,W,3) Image.
                gaussian_min  (int)        : Random gaussian kernel minimum size
                gaussian_max  (int)        : Random gaussian kernel maximum size
        """
        k = random.randrange(gaussian_min, gaussian_max, 2)    
        rgb = cv2.GaussianBlur(rgb, (k,k), 0)

        return rgb


    def image_random_brightness(self, rgb: np.ndarray, scale_min: float = 0.5,
                                scale_max: float = 1.0, factor: int = 100) -> np.ndarray:
        """
            Random brightness function   
            Args:
                rgb        (np.ndarray) : (H,W,3) Image.
                scale_min  (float)      : Minimum change in random blurring
                scale_max  (float)      : Maximum change in random blurring
                factor     (int)        : The maximum intensity of blurring,
                                          multiplied by a value between scale_min and scale_max.
        """
        random_val = np.random.uniform(low=scale_min, high=scale_max)
        factor *= random_val
        array = np.full(rgb.shape, (factor, factor, factor), dtype=np.uint8)

        return cv2.add(rgb, array)

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

        rows, cols = rgb.shape[:2]
        trans_mat = np.float64([[1, 0, random_dx], [0, 1, random_dy]])


        trans_rgb = cv2.warpAffine(rgb, trans_mat, (cols, rows))
        trans_mask = cv2.warpAffine(mask, trans_mat, (cols, rows))

        return trans_rgb, trans_mask

    def image_random_rotation(self, rgb: np.ndarray, mask: np.ndarray,
                              rot_min: int = 10, rot_max: int = 45) -> Union[np.ndarray, np.ndarray]:
        """
            Random rotation function   
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                mask     (np.ndarray) : (H,W,1) Image.
                rot_min  (int)        : Minimum rotation angle (degree).
                rot_max  (int)        : Maximum rotation angle (degree).
        """
        rot = random.randint(rot_min, rot_max)
        reverse = random.randint(1, 2)
        
        
        if reverse == 2:
            rot *= -1
        radian = rot * math.pi / 180.0 

        rgb = tfa.image.rotate(images=rgb, angles=radian).numpy()
        mask = tfa.image.rotate(images=mask, angles=radian).numpy()
        
        return rgb, mask


    def image_random_crop(self, rgb: np.ndarray, mask: np.ndarray,
                          aspect_ratio: float = 1.77) -> Union[np.ndarray, np.ndarray]:
        """
            Random crop function   
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                mask     (np.ndarray) : (H,W,1) Image.
                aspect_ratio  (float)        : Image Aspect Ratio. Code is written vertically.
        """
        widht_scale = tf.random.uniform([], 0.7, 0.95)
        
        new_w = rgb.shape[1] * widht_scale
        new_h = new_w * aspect_ratio
        
        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, axis=-1)
        
        concat_img = tf.concat([rgb, mask], axis=-1)
        concat_img = tf.image.random_crop(concat_img, size=[new_h, new_w, 4])
        
        crop_img = concat_img[:, :, :3].numpy()
        crop_mask = concat_img[:, :, 3:].numpy()

        return crop_img, crop_mask



    def plot_images(self, rgb: np.ndarray, mask: np.ndarray):
        """
            Image and mask plotting on screen function.
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                mask     (np.ndarray) : (H,W,1) Image.
                aspect_ratio  (float)        : Image Aspect Ratio. Code is written vertically.
        """
        rows = 1
        cols = 3

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        # convert to unit8 type
        rgb = rgb.astype(np.uint8)
        mask = mask.astype(np.uint8)

        fig = plt.figure()
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(rgb)
        ax0.set_title('img')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(mask)
        ax0.set_title('mask')
        ax0.axis("off")

        mask = tf.concat([mask, mask, mask], axis=-1)

        draw_mask = tf.where(mask >= 1, mask, rgb)

        ax0 = fig.add_subplot(rows, cols, 3)
        ax0.imshow(draw_mask)
        ax0.set_title('draw_mask')
        ax0.axis("off")
        plt.show()
        plt.close()


    def change_image(self, rgb: np.ndarray, obj_mask: np.ndarray) -> np.ndarray:
        """
            Change the part corresponding to the mask area in the RGB image.
            Various augmentation applications such as image synthesis and color conversion are possible.
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                obj_mask     (np.ndarray) : (H,W,1) Image.
        """

        contour_idx = 0

        obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_RGB2GRAY)
        obj_mask = obj_mask.astype(np.uint8)

        obj_idx = np.unique(obj_mask)
        obj_idx = np.delete(obj_idx, 0)

        for idx in obj_idx:  # 1 ~ obj nums
            binary_mask = np.where(obj_mask == idx, 255, 0)

            binary_mask = binary_mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour_idx += 1
                # 마스크의 contour를 이용하여 합성 할 영역의 bounding box를 계산
                # Calculate the bounding box of the area to be synthesized using the contour of the mask.
                x, y, w, h = cv2.boundingRect(contour)

                copy_rgb = rgb.copy()

                copy_rgb = self.image_random_bluring(rgb=copy_rgb)
                copy_rgb = self.image_random_brightness(rgb=copy_rgb)
                copy_rgb = self.image_histogram_equalization(rgb=copy_rgb)

                binary_mask_copy = binary_mask.copy()
                binary_mask_copy = np.expand_dims(binary_mask_copy, axis=-1)

                copy_mask = np.where(
                    binary_mask_copy[y:y+h, x:x+w] == 255, copy_rgb[y:y+h, x:x+w], rgb[y:y+h, x:x+w])

                rgb[y:y+h, x:x+w] = copy_mask

        return rgb


    def save_images(self, rgb: np.ndarray, mask: np.ndarray, prefix: str):
        """
            Save image and mask
            Args:
                rgb     (np.ndarray) : (H,W,3) Image.
                mask    (np.ndarray) : (H,W,1) Image.
                prefix  (str)        : The name of the image to be saved.
        """
        random_prefix_idx = random.randint(0, 10000)
        prefix = str(random_prefix_idx) + '_' + prefix

        for idx in range(len(self.label_list)):

            pixel_rgb = self.label_list[idx]['rgb']
            pixel_value = pixel_rgb[2] # BGR
            mask = np.where(mask==pixel_value, self.label_list[idx]['class_idx'], mask)
            
        visual_mask = mask * int(255 / len(self.label_list))

        cv2.imwrite(self.OUT_RGB_PATH + prefix +'_.png', rgb)
        cv2.imwrite(self.OUT_MASK_PATH + prefix + '_mask.png', mask)
        cv2.imwrite(self.OUT_VIS_MASK_PATH + prefix + '_visual_mask.png', visual_mask)

                                                              
if __name__ == '__main__':
    """
    Image augmentation can be selected according to the option using the internal function of ImageAugmentationLoader.
    """
    from tqdm import tqdm

    image_loader = ImageAugmentationLoader(args=args)
    rgb_list = image_loader.get_rgb_list()
    mask_list = image_loader.get_mask_list()
    obj_mask_list = image_loader.get_obj_mask_list()
    bg_list = image_loader.get_bg_list()


    # for idx in range(len(rgb_list)):
    for idx in tqdm(range(len(rgb_list)), total=len(rgb_list)):
        original_rgb = cv2.imread(rgb_list[idx])
        original_mask = cv2.imread(mask_list[idx])
        original_obj_mask = cv2.imread(obj_mask_list[idx])

        original_mask = original_mask[:, :, :1]
        
        
        rgb, mask, obj_mask = image_loader.image_resize(
            rgb=original_rgb, mask=original_mask, obj_mask=original_obj_mask, size=(1600, 900))
    
        # 1. Default image (1) + Random rotation (1) + Random translation (1)
        image_loader.save_images(rgb=rgb.copy(), mask=mask.copy(), prefix='idx_{0}_original_'.format(idx))

        rot_rgb, rot_mask = image_loader.image_random_rotation(rgb=rgb.copy(), mask=mask.copy())
        image_loader.save_images(rgb=rot_rgb, mask=rot_mask, prefix='idx_{0}_randomRot_'.format(idx))

        trans_rgb, trans_mask = image_loader.image_random_translation(rgb=rgb.copy(), mask=mask.copy(), min_dx=50, min_dy=50,
                                                                      max_dx=300, max_dy=400)
        image_loader.save_images(rgb=trans_rgb, mask=trans_mask, prefix='idx_{0}_randomTrans_'.format(idx))
        

        # 2. Image Color augmentation (1)
        color_aug_rgb = image_loader.image_random_bluring(rgb=rgb.copy())
        color_aug_rgb = image_loader.image_random_brightness(rgb=color_aug_rgb)
        image_loader.save_images(rgb=color_aug_rgb, mask=mask.copy(), prefix='idx_{0}_colorAug_'.format(idx))


        # 3. Image Random warpAffine
        for aff_idx in range(5):
            warp_rgb = rgb.copy()
            warp_mask = mask.copy()
            if tf.random.uniform([]) > 0.5:
                warp_rgb, warp_mask = image_loader.image_random_rotation(rgb=warp_rgb, mask=warp_mask)
            if tf.random.uniform([]) > 0.5:
                warp_rgb, warp_mask = image_loader.image_random_crop(rgb=warp_rgb, mask=warp_mask, aspect_ratio=1.33)
            if tf.random.uniform([]) > 0.5:
                warp_rgb, warp_mask = image_loader.image_random_translation(rgb=warp_rgb, mask=warp_mask, min_dx=50, min_dy=50,
                                                                      max_dx=300, max_dy=400)

            image_loader.save_images(rgb=warp_rgb, mask=warp_mask, prefix='idx_{0}_RanWarpAffine{1}_'.format(idx, aff_idx))
            

        # 4. Random rotation + crop (2)1
        for rot_crop_idx in range(3):
            rot_rgb, rot_mask = image_loader.image_random_rotation(rgb=rgb.copy(), mask=mask.copy())
            crop_rgb, crop_mask = image_loader.image_random_crop(rgb=rot_rgb, mask=rot_mask, aspect_ratio=1.33)
            image_loader.save_images(rgb=crop_rgb, mask=crop_mask, prefix='idx_{0}_RanRotCrop_{1}_'.format(idx, rot_crop_idx))
            
        

        # 5. Label area color jitter
        if len(np.delete(np.unique(np.mean(obj_mask.copy(), axis=-1)),0)) != 0: # Zero mask가 아닌 경우에만
            for change_label_area_idx in range(2):
                change_rgb = image_loader.change_image(rgb=rgb.copy(), obj_mask=obj_mask.copy())
                image_loader.save_images(rgb=change_rgb, mask=mask.copy(), prefix='idx_{0}_change_bg_{1}_'.format(idx, change_label_area_idx))