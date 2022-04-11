import numpy as np
import cv2
from imageio import imread, imwrite
import glob
import os
import argparse
import random
from utils import load_imgs, canny_edge, find_contours
import natsort
import re 
import math
from pathlib import Path
from tensorflow.keras.preprocessing.image import random_rotation

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_labeling/data/img/0405/040523_24cm_gray_d3_noExposure/result/semantic_label_mask_result/semantic_mask/input/')
parser.add_argument("--mask_path",     type=str,   help="raw image path", default='./data_labeling/data/img/0405/040523_24cm_gray_d3_noExposure/result/semantic_label_mask_result/semantic_mask/check_gt/')
parser.add_argument("--result_path",     type=str,   help="raw image path", default='./data_labeling/data/img/0405/040523_24cm_gray_d3_noExposure/result/semantic_label_mask_result/semantic_mask/augmentation/')

parser.add_argument("--bg_path",     type=str,   help="raw image path", default='./data_labeling/data/img/backgrounds/rgb/')

args = parser.parse_args()
RGB_PATH = args.rgb_path
MASK_PATH = args.mask_path
BG_PATH = args.bg_path
OUTPUT_PATH = args.result_path


os.makedirs(OUTPUT_PATH, exist_ok=True)

OUT_RGB_PATH = OUTPUT_PATH + 'rgb/'

OUT_MASK_PATH = OUTPUT_PATH + 'mask/'
os.makedirs(OUT_RGB_PATH, exist_ok=True)

os.makedirs(OUT_MASK_PATH, exist_ok=True)

rgb_list = glob.glob(os.path.join(RGB_PATH+'*.png'))
rgb_list = natsort.natsorted(rgb_list,reverse=True)

mask_list = glob.glob(os.path.join(MASK_PATH+'*.png'))
mask_list = natsort.natsorted(mask_list,reverse=True)

bg_list =  glob.glob(os.path.join(BG_PATH,  '*.png'))
bg_list = natsort.natsorted(bg_list,reverse=True)

def img_shift(rgb, mask, name):
    rand_x = random.randint(0, 100)
    rand_y = random.randint(0, 100)

    shift_x = rand_x
    shift_y = rand_y
    h, w = rgb.shape[:2]

    # plus affine
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aff_rgb_p = cv2.warpAffine(rgb, M, (w, h))
    # aff_depth_p = cv2.warpAffine(depth, M, (w, h))
    aff_mask_p = cv2.warpAffine(mask, M, (w, h))

    cv2.imwrite(OUT_RGB_PATH + name + '_p_shift' +'_.png', aff_rgb_p)
    # imwrite(OUT_DEPTH_PATH + name + '_p_shift'+'_depth.tif', aff_depth_p)
    cv2.imwrite(OUT_MASK_PATH +name + '_p_shift' +'_mask.png', aff_mask_p)

    M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    aff_rgb_m = cv2.warpAffine(rgb, M, (w, h))
    # aff_depth_m = cv2.warpAffine(depth, M, (w, h))
    aff_mask_m = cv2.warpAffine(mask, M, (w, h))

    cv2.imwrite(OUT_RGB_PATH + name + '_m_shift' +'_.png', aff_rgb_m)
    # imwrite(OUT_DEPTH_PATH + name + '_m_shift'+'_depth.tif', aff_depth_m)
    cv2.imwrite(OUT_MASK_PATH +name + '_m_shift' +'_mask.png', aff_mask_m)

    print('save_shift')
    # minus affine

def img_blur(img, mask, name):
    k = random.randrange(3,21,2)
    img = cv2.GaussianBlur(img, (k,k), 0)
    # depth = cv2.GaussianBlur(depth, (k,k), 0)
    # mask = cv2.GaussianBlur(mask, (k,k), 0)
    
    cv2.imwrite(OUT_RGB_PATH + name + '_blur' +'_.png', img)
    # imwrite(OUT_DEPTH_PATH + name + '_blur'+'_depth.tif', depth)
    cv2.imwrite(OUT_MASK_PATH +name + '_blur' +'_mask.png', mask)

def img_rotate(img, mask, name):
    rot = random.randint(10, 45)
    
    h, w = img.shape[:2]

    p_m = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
    m_m = cv2.getRotationMatrix2D((w/2, h/2), -rot, 1)

    rot_img_p = cv2.warpAffine(img, p_m, (w, h))
    # rot_depth_p = cv2.warpAffine(depth, p_m, (w, h))
    rot_mask_p = cv2.warpAffine(mask, p_m, (w, h))


    cv2.imwrite(OUT_RGB_PATH + name + '_rot_p' +'_.png', rot_img_p)
    # imwrite(OUT_DEPTH_PATH + name + '_rot_p'+'_depth.tif', rot_depth_p)
    cv2.imwrite(OUT_MASK_PATH +name + '_rot_p' +'_mask.png', rot_mask_p)

    rot_img_m = cv2.warpAffine(img, m_m, (w, h))
    # rot_depth_m = cv2.warpAffine(depth, m_m, (w, h))
    rot_mask_m = cv2.warpAffine(mask, m_m, (w, h))

    cv2.imwrite(OUT_RGB_PATH + name + '_rot_m' +'_.png', rot_img_m)
    # imwrite(OUT_DEPTH_PATH + name + '_rot_m'+'_depth.tif', rot_depth_m)
    cv2.imwrite(OUT_MASK_PATH +name + '_rot_m' +'_mask.png', rot_mask_m)

def cutmix(img, mask, bg, name):
    bg = cv2.resize(bg, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    rand_x = random.randint(0, 200)
    rand_y = random.randint(0, 200)

    shift_x = rand_x
    shift_y = rand_y
    h, w = img.shape[:2]

    # plus affine
    if random.randint(0, 1) == 1:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    else:
        M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    img = cv2.warpAffine(img, M, (w, h))
    # aff_depth_p = cv2.warpAffine(depth, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))
    
    # mask = mask[:, :, 0]
    binary_mask = np.where(mask >= 200, 1, 0).astype(np.uint8)
    
    img *= binary_mask


    bg_background = bg * np.where(binary_mask == 1, 0, 1).astype(np.uint8)
    bg_background += img

    cv2.imwrite(OUT_RGB_PATH + name + '_bg' +'_.png', bg_background)
    cv2.imwrite(OUT_MASK_PATH +name + '_bg' +'_mask.png', mask)


def object_resize(img, mask, bg, name, img_idx):
    bg = cv2.resize(bg, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 1. Resize 할 random factor값 가져오기
    rand_factor = round(random.uniform(1,2), 1)          
    rand_x = rand_factor
    rand_y = rand_factor

    # 2. 배경 rgb 이미지, zero mask 준비하기
    bg_img = bg.copy()
    zero_mask = np.zeros(bg.shape)
    zero_mask = zero_mask[:, :, 0]

    # 3. 이미지 리사이즈
    img = cv2.resize(img, dsize=(0, 0), fx=rand_x, fy=rand_y, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, dsize=(0, 0), fx=rand_x, fy=rand_y, interpolation=cv2.INTER_NEAREST)

    # 4. 랜덤 로테이션
    rot = random.randint(10, 350)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
    
    img = cv2.warpAffine(img, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))

    # 5. Binary 마스크 생성
    binary_mask = np.where(mask.copy() >= 1, 255, 0).astype(np.uint8)
    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    # 6. mask를 이용하여 object의 x,y,w,h 계산
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_allowed_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # if area >= 1000: 
        area_allowed_contour.append(contour)
    try:
        x,y,w,h = cv2.boundingRect(area_allowed_contour[0])
    except:
        return 0

    if x % 2 == 1:
        x += 1
    if y % 2 == 1:
        y += 1
    if w % 2 == 1:
        w += 1
    if h % 2 == 1:
        h += 1

    # 7. 좌표값을 이용하여 원본 이미지에서 crop

    crop_mask = mask.copy()[y:y+h, x:x+w]

    crop_rgb = img.copy()[y:y+h, x:x+w]
    crop_rgb *= np.where(crop_mask.copy() >= 1, 1, 0).astype(np.uint8)
    
    
    draw_mask = mask.copy()
    draw_mask = draw_mask[:, :, 0]
    draw_mask = draw_mask[y:y+h, x:x+w]
    
    
    bg_y, bg_x, _ = bg_img.shape
    """
    1280x720 
    if rand_factor > 1.0:
        x_max_shift = 250
        y_max_shift = 50
    else:
        x_max_shift = 500
        y_max_shift = 150
    """
    if rand_factor > 1.0:
        x_max_shift = 100
        y_max_shift = 20
    else:
        x_max_shift = 250
        y_max_shift = 50



    
    rand_bg_x = random.randint(0, x_max_shift)
    rand_bg_y = random.randint(0, y_max_shift)

    use_minus = random.randint(0, 1)
    if use_minus == 1:
        rand_bg_x *= -1
        rand_bg_y *= -1
    
    bg_y = ( bg_y // 2 ) + rand_bg_y
    bg_x = ( bg_x // 2 ) + rand_bg_x
    

    # crop image
    crop_y , crop_x, _ = crop_rgb.shape
    w = crop_x // 2# w // 2
    h = crop_y // 2# h // 2 
    
    # 8. Crop 이미지를 사용하여 배경에 합성

    bg_img[bg_y - h : bg_y + h, bg_x - w : bg_x + w] = np.where(crop_mask.copy() >=1, crop_rgb, bg_img[bg_y - h : bg_y + h, bg_x - w : bg_x + w])
    hole_label = zero_mask.copy()

    zero_mask[bg_y - h : bg_y + h, bg_x - w : bg_x + w] = np.where(draw_mask.copy() >= 127, 1, 0)
    zero_mask = zero_mask.astype(np.uint8)
    hole_label[bg_y - h : bg_y + h, bg_x - w : bg_x + w] = np.where(draw_mask.copy() >= 250, 1, 0)
    hole_label = hole_label.astype(np.uint8)

    zero_mask += hole_label
    
    # 9. 이미지 저장
    # cv2.imshow('bg_img', bg_img)
    # cv2.waitKey(0)
    # print(bg_img.shape)
    # cv2.imshow('zero_mask', zero_mask * 127)
    # cv2.waitKey(0)
    
    img_name = '_factor_' + str(rand_factor) + '_' + str(img_idx)
    cv2.imwrite(OUT_RGB_PATH + name + img_name +'_original' +'_.png', bg_img)
    
    cv2.imwrite(OUT_MASK_PATH +name +  img_name +'_synthesis' +'_mask.png', zero_mask * 127)


def save_imgs(rgb, mask, name):
    cv2.imwrite(OUT_RGB_PATH + name + '_original' +'_.png', rgb)
    # imwrite(OUT_DEPTH_PATH + name + '_original'+'_depth.tif', depth)
    cv2.imwrite(OUT_MASK_PATH +name + '_original' +'_mask.png', mask)
    print('save_original')


if __name__ == '__main__': 
    print(rgb_list[0], mask_list[0])
    i = 1

    bg_range = range(len(bg_list))
    
    for idx in range(len(rgb_list)):
        img = cv2.imread(rgb_list[idx])
        mask = cv2.imread(mask_list[idx])

        
        
        name = rgb_list[idx].split('/')[5] + '_' + str(i)
        
        save_imgs(img, mask, name)
        img_shift(img, mask, name)
        img_blur(img, mask, name)
        img_rotate(img, mask, name)
        
        for bg_idx in bg_range:
            bg_img = cv2.imread(bg_list[bg_idx])
            for j in range(5):
                object_resize(img, mask, bg_img, name, j)
        i+=1

        
        
        

        
        
