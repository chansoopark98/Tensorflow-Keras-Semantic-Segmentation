import json
import cv2
import glob
import natsort
import os
import numpy as np
import PIL

RAW_PATH = './data_augmentation/raw_data/labelme_img/'
SAVE_RGB_PATH = RAW_PATH + 'rgb/'
SAVE_MASK_PATH = RAW_PATH + 'mask/'

os.makedirs(SAVE_RGB_PATH, exist_ok=True)
os.makedirs(SAVE_MASK_PATH, exist_ok=True)

rgb_list = glob.glob(os.path.join(RAW_PATH+'*.png'))
rgb_list = natsort.natsorted(rgb_list,reverse=True)

json_list = glob.glob(os.path.join(RAW_PATH+'*.json'))
json_list = natsort.natsorted(json_list)

save_idx = 0
for json_sample in json_list:
    with open(json_sample, "r") as read_file:
        data = json.load(read_file)

        
        image = cv2.imread(RAW_PATH + data['imagePath'])

        points_arrays = data['shapes'][0]['points']

        point_x = []
        point_y = []

        for point_idx in range(len(points_arrays)):
            
            x = points_arrays[point_idx][0]
            y = points_arrays[point_idx][1]

            point_x.append(x)
            point_y.append(y)

        ab = np.stack((point_x, point_y), axis=1)
        ab = np.int32(ab)
        

        mask = np.zeros((image.shape[0],image.shape[1]))
        
        mask = cv2.fillPoly(mask, pts = [ab], color =(255))
        mask = np.where(mask==255, 1, 0).astype(np.uint8)

        cv2.imwrite(SAVE_RGB_PATH + 'human_dataset1_{0}_image.jpg'.format(save_idx), image)
        cv2.imwrite(SAVE_MASK_PATH + 'human_dataset1_{0}_mask.png'.format(save_idx), mask)

        save_idx += 1