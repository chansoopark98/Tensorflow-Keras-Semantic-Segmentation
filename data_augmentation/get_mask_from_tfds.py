import tensorflow_datasets as tfds
import cv2
import argparse
import os
import numpy as np

train_data = tfds.load('nyu_depth_v2', data_dir='./datasets/', split='train')

parser = argparse.ArgumentParser()
parser.add_argument("--output_path",     type=str,   help="Path to save the conversion result", default='./data_augmentation/raw_data/nyu_depth/select/')

args = parser.parse_args()

class ImageAugmentationLoader():
    def __init__(self, args):
        
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'gt/'
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_MASK_PATH, exist_ok=True)

idx = 0
generator = ImageAugmentationLoader(args)
for sample in train_data.take(10000):
    idx += 1
    image = sample['image'].numpy()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.zeros(image.shape[:2])
    
    cv2.imwrite(generator.OUT_RGB_PATH + 'bg_image_{0}'.format(idx) +'_rgb.jpg', image)
    cv2.imwrite(generator.OUT_MASK_PATH + 'bg_image_{0}'.format(idx) + '_mask.png', mask)
