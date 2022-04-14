import argparse
import os
import glob
import cv2
import numpy as np
import natsort
from data_labeling.utils import canny_selector
        
parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_labeling/data/img/0321/032110_white_r1_40cm/result/rgb/') 
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_labeling/data/img/0321/032110_white_r1_40cm/result/mask/')
parser.add_argument("--result_path",     type=str,   help="raw image path", default='./data_labeling/data/img/0321/032110_white_r1_40cm/result/semantic_label')

args = parser.parse_args()
RGB_PATH = args.rgb_path
MASK_PATH = args.mask_path
RESULT_DIR = args.result_path
MASK_RESULT_DIR = RESULT_DIR + '_mask_result/'

SEMANTIC_PATH = MASK_RESULT_DIR + 'semantic_mask/'
SEMANTIC_INPUT_PATH = SEMANTIC_PATH + 'input/'
SEMANTIC_GT_PATH = SEMANTIC_PATH + 'gt/'
SEMANTIC_CHECK_GT_PATH = SEMANTIC_PATH + 'check_gt/'

os.makedirs(MASK_RESULT_DIR, exist_ok=True)
os.makedirs(SEMANTIC_PATH, exist_ok=True)
os.makedirs(SEMANTIC_INPUT_PATH, exist_ok=True)
os.makedirs(SEMANTIC_GT_PATH, exist_ok=True)
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
        # if area >= 1000: 
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

    draw_result = result.copy()

    canny = cv2.GaussianBlur(ROI.copy(), (7, 7), 0)
    canny = canny_selector(canny)


    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        
        circle_contour.append(contour)

    cv2.namedWindow("circle check") 
    for i in range(len(circle_contour)):
        draw_roi = ROI.copy()
        cv2.drawContours(draw_roi, circle_contour, i, (0, 255, 0), -1)
        cv2.moveWindow("circle check", 800, 400)
        cv2.imshow('circle check', draw_roi)
        
        key = cv2.waitKey(0)

        delete_idx = abs(48 - key)
        
        if delete_idx == 65:
            cv2.destroyAllWindows()
            break
                
        # 1번 키를 누를 때
        if key == 49:
            cv2.destroyAllWindows()
            draw_canny = canny.copy()
            cv2.drawContours(draw_canny, circle_contour, i, 127, -1)
            cropped_gt = np.where(draw_canny==127, 1, 0)
            draw_result[y:y+h, x:x+w] += cropped_gt.astype(np.uint8)
            break
    
    cv2.namedWindow("check gt")
    cv2.moveWindow("check gt", 600, 400)
    cv2.imshow('check gt', draw_result * 127)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    delete_idx = abs(48 - key)
            
    # 1번 키를 누를 때
    if key == 49:
        cv2.destroyAllWindows()

        print('save : ', file_idx)
        cv2.imwrite(SEMANTIC_INPUT_PATH +str(file_idx) +'_rgb.png', original)
        cv2.imwrite(SEMANTIC_GT_PATH +str(file_idx) +'_semantic_mask.png', draw_result )
        cv2.imwrite(SEMANTIC_CHECK_GT_PATH +str(file_idx) +'_semantic_mask.png', draw_result * 127)
        file_idx += 1
