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

# from utils.cityscape_colormap import class_weight
# from utils.adamW import LearningRateScheduler, poly_decay
# import tensorflow_addons
# sudo apt-get install libtcmalloc-minimal4
# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py
# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py
# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python gan_train.py


tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--result_path",     type=str,   help="test result path", default='./test_imgs/')
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=100)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--optimizer",     type=str,   help="Optimizer", default='adam')
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--result_dir", type=str,   help="Test result dir", default='./results/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=False)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=True)
parser.add_argument("--distribution_mode",  type=bool,  help="분산 학습 모드 설정", default=True)

args = parser.parse_args()
RESULT_PATH = args.result_path
WEIGHT_DECAY = args.weight_decay
OPTIMIZER_TYPE = args.optimizer
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
RESULT_DIR = args.result_dir
MASK_RESULT_DIR = RESULT_DIR + 'mask_result/'
IMAGE_SIZE = (480, 640)
# IMAGE_SIZE = (None, None)
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode

ROI_PATH = MASK_RESULT_DIR + 'roi_mask/'
ROI_INPUT_PATH = ROI_PATH + 'input/'
ROI_GT_PATH = ROI_PATH + 'gt/'
ROI_CHECK_GT_PATH = ROI_PATH + 'check_gt/'

SEMANTIC_PATH = MASK_RESULT_DIR + 'semantic_mask/'
SEMANTIC_INPUT_PATH = SEMANTIC_PATH + 'input/'
SEMANTIC_GT_PATH = SEMANTIC_PATH + 'gt/'
SEMANTIC_CHECK_GT_PATH = SEMANTIC_PATH + 'check_gt/'

if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MASK_RESULT_DIR, exist_ok=True)

DEBUG_RESULT_DIR = RESULT_DIR +'debug_result/'
os.makedirs(DEBUG_RESULT_DIR, exist_ok=True)

dataset_config = DatasetGenerator(DATASET_DIR, (480, 640), BATCH_SIZE, mode='all')
data = dataset_config.get_testData(dataset_config.data)

model = segmentation_model(image_size=IMAGE_SIZE)


weight_name = '_0323_L-bce_B-16_E-100_Optim-Adam_best_iou'

# weight_name = '_0318_final_loss'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

model.summary()
i = 1

for x, gt, original in data.take(dataset_config.number_all):
    gt = gt.numpy()[0, :, :, 0]
    cv2.imwrite(DEBUG_RESULT_DIR+'gt.png', gt)
    original = original.numpy()[0]
    cv2.imwrite(DEBUG_RESULT_DIR+'rgb.png', original)
    gray_scale = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    
                                    
    pred = model.predict_on_batch(x)
    pred = np.where(pred>=1.0, 1, 0)
    result = pred[0]
    

    result= result[:, :, 0].astype(np.uint8)
    result_mul = result.copy() * 255
    cv2.imwrite(DEBUG_RESULT_DIR+'dl_pred.png', result_mul)
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
    cv2.imwrite(DEBUG_RESULT_DIR+'bbox_roi.png', ROI)
    ROI = cv2.resize(ROI, dsize=(w *4, h*4), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(DEBUG_RESULT_DIR+'resized bbox_roi.png', ROI)
    ROI = cv2.GaussianBlur(ROI, (3, 3), 0)
    cv2.imwrite(DEBUG_RESULT_DIR+'blur resized bbox_roi.png', ROI)
    # _, ROI = cv2.threshold(ROI,100,255,cv2.THRESH_BINARY)
    _, ROI = cv2.threshold(ROI,200,255,cv2.THRESH_BINARY)
    cv2.imwrite(DEBUG_RESULT_DIR+'binary threshold.png', ROI)
    
    zero_img = np.zeros(gray_scale.shape)
    zero_ROI = np.zeros(ROI.shape)

    contours, _ = cv2.findContours(ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 999999
    
    out_contour = []
    i = 0
    draw_contours = ROI.copy()
    draw_contours = np.expand_dims(draw_contours, axis=-1)
    draw_contours = np.concatenate([draw_contours, draw_contours ,draw_contours], axis =-1)
    for contour in contours:
        cv2.drawContours(draw_contours, contours, i, (0, 0, 255), 2)
        
        area = cv2.contourArea(contour)
        i += 1
        if min_area >= area:
            min_area = area
            out_contour = [contour]
    cv2.imwrite(DEBUG_RESULT_DIR+'draw_contours.png', draw_contours)
    draw_img = ROI.copy()
    cv2.drawContours(draw_img, out_contour, 0, (127, 127, 127), -1)       
    cv2.imwrite(DEBUG_RESULT_DIR+'draw contour.png', draw_img)

    draw_img = cv2.resize(draw_img, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(DEBUG_RESULT_DIR+'resized draw contour.png', draw_img)
    draw_img = np.where(draw_img==127, 2, draw_img)
    draw_img = np.where(draw_img==255, 1, draw_img)

    

    cropped_gt = draw_img.copy()
    cropped_gt = np.where(cropped_gt==2, 1, 0)
    gt[y:y+h, x:x+w] += cropped_gt
    cv2.imwrite(DEBUG_RESULT_DIR+'final.png', gt * 127)
    break