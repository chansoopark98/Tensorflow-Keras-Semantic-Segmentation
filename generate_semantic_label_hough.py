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
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040829_exposure_1000_gain_100_25cm_gray3/result/rgb/')
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040829_exposure_1000_gain_100_25cm_gray3/result/mask/')

parser.add_argument("--result_path",     type=str,   help="raw image path", default='./data_labeling/data/img/040829_exposure_1000_gain_100_25cm_gray3/result/semantic_label')

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

file_idx = 1
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
            

    
    # cv2.drawContours(draw_img, out_contour, 0, (127, 127, 127), -1)

    draw_result = result.copy()

    cv2.namedWindow("Trackbar Windows")

    cv2.createTrackbar("minDistance", "Trackbar Windows", 0, 255, lambda x : x)
    cv2.createTrackbar("CannyThreshold", "Trackbar Windows", 0, 255, lambda x : x)
    cv2.createTrackbar("CenterThreshold", "Trackbar Windows", 0, 255, lambda x : x)
    cv2.createTrackbar("minRadius", "Trackbar Windows", 0, 255, lambda x : x)
    cv2.createTrackbar("maxRadius", "Trackbar Windows", 0, 255, lambda x : x)

    """
    gray

    minDistance : 5
    CannyThreshold : 15
    CenterThreshold : 25
    minRadius : 27
    maxRadius : 30
    """
    cv2.setTrackbarPos("minDistance", "Trackbar Windows", 5)
    cv2.setTrackbarPos("CannyThreshold", "Trackbar Windows", 15)
    cv2.setTrackbarPos("CenterThreshold", "Trackbar Windows", 17)
    cv2.setTrackbarPos("minRadius", "Trackbar Windows", 4)
    cv2.setTrackbarPos("maxRadius", "Trackbar Windows", 12)

    while cv2.waitKey(1) != ord('q'):
        draw_img = ROI.copy()
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
        try:
            minDistance = cv2.getTrackbarPos("minDistance", "Trackbar Windows")
            CannyThreshold = cv2.getTrackbarPos("CannyThreshold", "Trackbar Windows")
            CenterThreshold = cv2.getTrackbarPos("CenterThreshold", "Trackbar Windows")
            minRadius = cv2.getTrackbarPos("minRadius", "Trackbar Windows")
            maxRadius = cv2.getTrackbarPos("maxRadius", "Trackbar Windows")

        except:
            print('out of range track bars')

            minDistance =5
            CannyThreshold = 15
            CenterThreshold =25
            minRadius = 27
            maxRadius = 30



        circles = cv2.HoughCircles(draw_img, cv2.HOUGH_GRADIENT, 1, minDistance,
                param1=CannyThreshold, param2=CenterThreshold, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            for i in circles[0]:
                cv2.circle(draw_img, (int(i[0]), int(i[1])), int(i[2]), 255, 1)
                cv2.circle(draw_img, (int(i[0]), int(i[1])), 0, 0, -1)
        cv2.imshow("Trackbar Windows", draw_img)
    

    draw_result *= 127
    if circles is not None:
        for i in circles[0]:
            new_roi = ROI.copy()
            
            cv2.circle(new_roi, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), -1)
            cv2.imshow('circle check', new_roi)
            key = cv2.waitKey(0)

            delete_idx = abs(48 - key)
            
            if delete_idx == 65:
                cv2.destroyAllWindows()
                break
                
                
                
            
            # 1번 키를 누를 때
            if key == 49:
                cv2.destroyAllWindows()
                
                
                
                cv2.circle(draw_result, (int(i[0] + x), int(i[1]) + y), int(i[2]), 255, -1)

                continue

            if key == ord('q'):
                break
        
        
        cv2.imshow('check gt', draw_result)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        delete_idx = abs(48 - key)

                
        # 1번 키를 누를 때
        if key == 49:
            cv2.destroyAllWindows()

            print('save : ', file_idx)
            cv2.imwrite(SEMANTIC_INPUT_PATH +str(file_idx) +'_rgb.png', original)
            cv2.imwrite(SEMANTIC_GT_PATH +str(file_idx) +'_semantic_mask.png', draw_result // 127)
            cv2.imwrite(SEMANTIC_CHECK_GT_PATH +str(file_idx) +'_semantic_mask.png', draw_result)
                
            file_idx += 1
        
    # # result = np.where(draw_result == 2, 2, result)

    # cv2.namedWindow("result")
    # cv2.moveWindow("result", 800, 400)
    # cv2.imshow('result', draw_result)
    # cv2.waitKey(0)
    

    # key = cv2.waitKey(0)
    
    
        
    # delete_idx = abs(48 - key)
    


                            
