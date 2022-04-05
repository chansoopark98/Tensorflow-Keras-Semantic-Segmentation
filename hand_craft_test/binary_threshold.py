from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import argparse
import time
import os
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

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]



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

if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MASK_RESULT_DIR, exist_ok=True)

batch_idx = 0
avg_duration = 0
ng_time = 0

img_list = glob.glob(os.path.join(RESULT_PATH,'*.png'))
img_list.sort()

def onChange(pos):
    pass

for i in range(len(img_list)):
    
    img = cv2.imread(img_list[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Trackbar Windows")
    cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
    cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)

    cv2.setTrackbarPos("threshold", "Trackbar Windows", 127)
    cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)

    while cv2.waitKey(1) != ord('q'):

        thresh = cv2.getTrackbarPos("threshold", "Trackbar Windows")
        maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")

        _, binary = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

        cv2.imshow("Trackbar Windows", binary)
    

print(f"avg inference time : {(avg_duration / len(img_list)) // 1000000}ms.")
print(f"No good images : {ng_time}.")