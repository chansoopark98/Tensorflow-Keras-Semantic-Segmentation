from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from models.model_builder import segmentation_model
from utils.load_datasets import DatasetGenerator
import argparse
import time
import os
import glob
import cv2
import numpy as np
import math
from pathlib import Path
import natsort
import re 

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])



parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040802_exposure_1000_gain_100_25cm_gray1/result/rgb/')
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040802_exposure_1000_gain_100_25cm_gray1/result/mask/')

parser.add_argument("--result_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040802_exposure_1000_gain_100_25cm_gray1/result/semantic_label')

args = parser.parse_args()
RGB_PATH = args.rgb_path
MASK_PATH = args.mask_path
RESULT_DIR = args.result_path
RESOLUTION = (640,480)


MASK_RESULT_DIR = RESULT_DIR + '_mask_result/'
IMAGE_SIZE = (480, 640)
# IMAGE_SIZE = (None, None)

ROI_PATH = MASK_RESULT_DIR + 'roi_mask/'

ROI_INPUT_PATH = ROI_PATH + 'input/'
ROI_GT_PATH = ROI_PATH + 'gt/'
ROI_CHECK_GT_PATH = ROI_PATH + 'check_gt/'

SEMANTIC_PATH = MASK_RESULT_DIR + 'semantic_mask/'

SEMANTIC_INPUT_PATH = SEMANTIC_PATH + 'input/'
SEMANTIC_GT_PATH = SEMANTIC_PATH + 'gt/'
SEMANTIC_CHECK_GT_PATH = SEMANTIC_PATH + 'check_gt/'


os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MASK_RESULT_DIR, exist_ok=True)

os.makedirs(ROI_PATH, exist_ok=True)
os.makedirs(ROI_INPUT_PATH, exist_ok=True)
os.makedirs(ROI_GT_PATH, exist_ok=True)

os.makedirs(SEMANTIC_PATH, exist_ok=True)
os.makedirs(SEMANTIC_INPUT_PATH, exist_ok=True)
os.makedirs(SEMANTIC_GT_PATH, exist_ok=True)

os.makedirs(ROI_CHECK_GT_PATH, exist_ok=True)
os.makedirs(SEMANTIC_CHECK_GT_PATH, exist_ok=True)

rgb_list = glob.glob(os.path.join(RGB_PATH+'*.png'))
rgb_list = natsort.natsorted(rgb_list,reverse=True)

mask_list = glob.glob(os.path.join(MASK_PATH+'*.png'))
mask_list = natsort.natsorted(mask_list,reverse=True)

i = 1
for idx in range(len(rgb_list)):
    img = cv2.imread(rgb_list[idx])

    gt = cv2.imread(mask_list[idx])
    mask = np.where(gt.copy()>= 200, 1.0 , 0)
    
    original = img
    gray_scale = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    
    result = mask

    result= result[:, :, 0].astype(np.uint8)
    result_mul = result.copy() * 255
    hh, ww = result_mul.shape

    contours, _ = cv2.findContours(result_mul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area >= 1000:
            circle_contour.append(contour)
            
    try:
        x,y,w,h = cv2.boundingRect(circle_contour[0])
    except:
        continue
        
    center_x = x + (w/2)
    center_y = y + (h/2)
    
    gray_scale *= result

    ROI = gray_scale.copy()[y:y+h, x:x+w]
    ROI = cv2.resize(ROI, dsize=(w *4, h*4), interpolation=cv2.INTER_LINEAR)
    ROI = cv2.GaussianBlur(ROI, (3, 3), 0)
    # _, ROI = cv2.threshold(ROI,100,255,cv2.THRESH_BINARY)
    _, ROI = cv2.threshold(ROI,200,255,cv2.THRESH_BINARY)
    
    circles = cv2.HoughCircles(ROI, cv2.HOUGH_GRADIENT, 1, 1,
                     param1=127, param2=1, minRadius=1, maxRadius=9)
    
    zero_img = np.zeros(gray_scale.shape)
    zero_ROI = np.zeros(ROI.shape)

    if circles is not None:
        cx, cy, radius = circles[0][0]
        contours, _ = cv2.findContours(ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 999999
        
        out_contour = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area >= area:
                min_area = area
                out_contour = [contour]
            

        draw_img = ROI.copy()
        cv2.drawContours(draw_img, out_contour, 0, (127, 127, 127), -1)
        cv2.imshow('draw_img', draw_img)
        cv2.waitKey(0)

        key = cv2.waitKey(0)
        print(key)
        cv2.destroyAllWindows()
            
        delete_idx = abs(48 - key)
            
        if delete_idx == 65:
            continue
            
        # try:
        # 1번 키를 누를 때
        if key == 49:
            print('save')
            save_draw_img = draw_img.copy()

            save_draw_img = np.where(save_draw_img == 255, 1, save_draw_img)
            save_draw_img = np.where(save_draw_img == 127, 2, save_draw_img)

            draw_img = cv2.resize(draw_img, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            draw_img = np.where(draw_img==127, 2, draw_img)
            draw_img = np.where(draw_img==255, 1, draw_img)

            cv2.imwrite(ROI_INPUT_PATH +str(i) +'_rgb.png', original[y:y+h, x:x+w])
            cv2.imwrite(ROI_GT_PATH +str(i) +'_semantic_mask.png', draw_img)
            cv2.imwrite(ROI_CHECK_GT_PATH +str(i) +'_semantic_mask.png', draw_img * 127)
            
            # zero_img[y:y+h, x:x+w] = draw_img
            cropped_gt = draw_img.copy()
            cropped_gt = np.where(cropped_gt==2, 1, 0)
            result[y:y+h, x:x+w] += cropped_gt.astype(np.uint8)
            cv2.imwrite(SEMANTIC_INPUT_PATH +str(i) +'_rgb.png', original)
            cv2.imwrite(SEMANTIC_GT_PATH +str(i) +'_semantic_mask.png', result)
            cv2.imwrite(SEMANTIC_CHECK_GT_PATH +str(i) +'_semantic_mask.png', result* 127)
                
            i += 1
