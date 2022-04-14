from cv2 import threshold
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from models.model_builder import segmentation_model
from utils.load_datasets import DatasetGenerator
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
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

def onMouse(event, x, y, flags, param): 
    # event = 10 휠 조절
    # event = 3 휠 클릭
    filled = 5
    
    
    if event==cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭 
    
    
        rgb_x = param[2]
        rgb_y = param[3]
        # param[0][y, x] = (0, 255, 0)
        param[1][rgb_y + y, rgb_x + x] = 2
        
        area = param[0][y- param[4] : y + param[4], x - param[4] : x + param[4]]

        if np.any(area == (0, 255, 0)):
            # param[0][y- param[4] : y + param[4], x - param[4] : x + param[4]] = (0, 255, 0)
            param[0][y- filled : y + filled, x - filled : x + filled] = (0, 255, 0)  
        
        new_v = np.where(abs(area - param[0][y, x]) <=(param[5], param[5] ,param[5]), (0, 255, 0), area)
        param[0][y- param[4] : y + param[4], x - param[4] : x + param[4]] = new_v
        

        
        # param[1][(rgb_y + y) - param[4]:(rgb_y + y) + param[4], (rgb_x + x) - param[4] : (rgb_x + x) + param[4]] = 2
        semantic_v = np.where(new_v == (0, 255, 0), 2, 0)
        param[1][(rgb_y + y) - param[4]:(rgb_y + y) + param[4], (rgb_x + x) - param[4] : (rgb_x + x) + param[4]] = semantic_v[:, :, 0]


    if event==3: # 휠 클릭
        rgb_x = param[2]
        rgb_y = param[3]

        # param[0][y- param[4] : y + param[4], x - param[4] : x + param[4]] = 0
        # param[1][(rgb_y + y) - param[4]:(rgb_y + y) + param[4], (rgb_x + x) - param[4] : (rgb_x + x) + param[4]] = 0
        param[0][y- filled : y + filled, x - filled : x + filled] = 0
        param[1][(rgb_y + y) - filled:(rgb_y + y) + filled, (rgb_x + x) - filled : (rgb_x + x) + filled] = 0

    
    cv2.imshow('draw_img',param[0])
    cv2.resizeWindow('draw_img', IMAGE_SIZE[1], IMAGE_SIZE[0])
        
parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_labeling/data/img/0321/032110_white_r1_40cm/result/rgb/')
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_labeling/data/img/0321/032110_white_r1_40cm/result/mask/')

parser.add_argument("--result_path",     type=str,   help="raw image path", default='./data_labeling/data/img/0321/032110_white_r1_40cm/result/semantic_label')

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
    
    rgb_map = original.copy()
    rgb_map = cv2.bitwise_and(rgb_map, rgb_map, mask=result)

    # >>>> ROI CROP
    ROI = rgb_map.copy()[y:y+h, x:x+w]
    # ROI = cv2.resize(ROI, dsize=(w * 4, h * 4), interpolation=cv2.INTER_LINEAR)
    # ROI = cv2.GaussianBlur(ROI, (3, 3), 0)
    # _, ROI = cv2.threshold(ROI,100,255,cv2.THRESH_BINARY)
    # _, ROI = cv2.threshold(ROI,200,255,cv2.THRESH_BINARY)
    
    # circles = cv2.HoughCircles(ROI, cv2.HOUGH_GRADIENT, 1, 1,
    #                  param1=127, param2=cv2.imshow('img',param[0]

    # if circles is not None:
    #     cx, cy, radius = circles[0][0]
    #     contours, _ = cv2.findContours(ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     min_area = 999999
        
    #     out_contour = []
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if min_area >= area:
    #             min_area = area
    #             out_contour = [contour]
            

    draw_img = ROI.copy()
    # cv2.drawContours(draw_img, out_contour, 0, (127, 127, 127), -1)

    draw_result = result.copy()

    cv2.namedWindow("draw_img")
    cv2.moveWindow("draw_img", 800, 400)

    cv2.createTrackbar("kernel_size", "draw_img", 1, 30, lambda x : x)
    cv2.setTrackbarPos("kernel_size", "draw_img", 13)
    
    cv2.createTrackbar("threshold", "draw_img", 1, 255, lambda x : x)
    cv2.setTrackbarPos("threshold", "draw_img", 5)
    

   
    while cv2.waitKey(1) != ord('q'):
        kernel_size = cv2.getTrackbarPos("kernel_size", "draw_img")
        pixel_threshold = cv2.getTrackbarPos("threshold", "draw_img")
        cv2.imshow('draw_img', draw_img)
        cv2.setMouseCallback('draw_img', onMouse,[draw_img, draw_result ,x, y, kernel_size, pixel_threshold])

    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    result = np.where(draw_result == 2, 2, result)

    cv2.namedWindow("result")
    cv2.moveWindow("result", 800, 400)
    cv2.imshow('result', result.copy() * 127)
    cv2.waitKey(0)
    

    key = cv2.waitKey(0)
    
    cv2.destroyAllWindows()
        
    delete_idx = abs(48 - key)
    

    # Choose don't save
    if delete_idx == 65:
        continue
        
    # Choose save
    # 1번 키를 누를 때
    if key == 49:
        print('save')
        cv2.imwrite(ROI_INPUT_PATH +str(i) +'_rgb.png', original[y:y+h, x:x+w])
        cv2.imwrite(ROI_GT_PATH +str(i) +'_semantic_mask.png', result[y:y+h, x:x+w])
        cv2.imwrite(ROI_CHECK_GT_PATH +str(i) +'_semantic_mask.png', result[y:y+h, x:x+w] * 127)
               
        cv2.imwrite(SEMANTIC_INPUT_PATH +str(i) +'_rgb.png', original)
        cv2.imwrite(SEMANTIC_GT_PATH +str(i) +'_semantic_mask.png', result)
        cv2.imwrite(SEMANTIC_CHECK_GT_PATH +str(i) +'_semantic_mask.png', result* 127)
            
        i += 1
