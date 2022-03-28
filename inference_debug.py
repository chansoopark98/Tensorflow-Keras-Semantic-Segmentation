from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from models.model_builder import base_model
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

from data_labeling.utils import canny_edge




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

if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MASK_RESULT_DIR, exist_ok=True)

model = base_model(image_size=IMAGE_SIZE)


weight_name = '_0323_L-bce_B-16_E-100_Optim-Adam_best_iou'

# weight_name = '_0318_final_loss'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

model.summary()
batch_idx = 0
avg_duration = 0

img_list = glob.glob(os.path.join(RESULT_PATH,'*.png'))
img_list.sort()

img = cv2.imread(img_list[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = tf.image.resize(img, size=IMAGE_SIZE,
                method=tf.image.ResizeMethod.BILINEAR)   
img = tf.cast(img, dtype=tf.float32)
img = preprocess_input(img, mode='torch')
img = tf.expand_dims(img, axis=0)
pred = model.predict_on_batch(img)

for i in range(len(img_list)):
    start = time.perf_counter_ns()
    img = cv2.imread(img_list[i])
    original = img.copy()
    gray_sclae = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, size=IMAGE_SIZE,
                    method=tf.image.ResizeMethod.BILINEAR)
                                    
    img = tf.cast(img, dtype=tf.float32)
    img = preprocess_input(img, mode='torch')
    img = tf.expand_dims(img, axis=0)
    pred = model.predict_on_batch(img)
    pred = np.where(pred>=1.0, 1, 0)
    result = pred[0]

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
        print('no holse')
        continue
    
    center_x = x + (w/2)
    center_y = y + (h/2)

    gray_sclae *= result

    ROI = gray_sclae.copy()[y:y+h, x:x+w]
    ROI = cv2.resize(ROI, dsize=(w *4, h*4), interpolation=cv2.INTER_LINEAR)

    _, ROI = cv2.threshold(ROI,100,255,cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(ROI, cv2.HOUGH_GRADIENT, 1, 1,
                     param1=50, param2=1, minRadius=1, maxRadius=10)
    
    zero_img = np.zeros(gray_sclae.shape)

    zero_ROI = np.zeros(ROI.shape)
    if circles is not None:
        cx, cy, radius = circles[0][0]
        cv2.circle(ROI, (int(cx), int(cy)), int(radius * 3), (0, 0, 0), 2, cv2.LINE_AA)
        zero_ROI[int(cy)-5:int(cy)+5, int(cx)-5:int(cx)+5] = 255
        
    ROI = cv2.resize(ROI, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    zero_ROI = cv2.resize(zero_ROI, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        
    zero_img[y:y+h, x:x+w] = zero_ROI
    
    yx_coords = np.mean(np.column_stack(np.where(zero_img == 255)),axis=0)
    
    duration = (time.perf_counter_ns() - start) / BATCH_SIZE
    avg_duration += duration

    if np.isnan(yx_coords[0]) != True:
        cv2.circle(original, (int(yx_coords[1]), int(yx_coords[0])), int(radius), (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('final result', original)
    cv2.waitKey(0)
    print(f"inference time : {duration // 1000000}ms.")
print(f"avg inference time : {(avg_duration / 1000) // 1000000}ms.")

    





    # if circles is not None:
    #     for i in  range(circles.shape[1]):
    #         cx, cy, radius = circles[0][0]
            # cv2.circle(dst, (int(cx), int(cy)), int(radius), (255, 255, 0), 2, cv2.LINE_AA)
    # cv2.circle(dst, (int(center_x), int(center_y)), int(radius), (255, 255, 0), 2, cv2.LINE_AA)
    # cv2.drawContours(dst, contours, 0, (127, 127, 127), 2)


# for i in range(1000):
#     start = time.perf_counter_ns()

#     img = cv2.imread('inference_test.png')
#     gray_sclae = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_sclae = cv2.GaussianBlur(gray_sclae, (0, 0), 1.0)
#     gray_sclae = cv2.resize(gray_sclae, dsize=(IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = tf.image.resize(img, size=IMAGE_SIZE,
#                     method=tf.image.ResizeMethod.BILINEAR)
                                    
#     img = tf.cast(img, dtype=tf.float32)
#     img = preprocess_input(img, mode='torch')
#     img = tf.expand_dims(img, axis=0)
#     pred = model.predict_on_batch(img)
#     result = pred[0]

#     result= result[:, :, 0].astype(np.uint8) 

#     gray_sclae *= result

#     circles = cv2.HoughCircles(gray_sclae, cv2.HOUGH_GRADIENT, 1, 1, param1=120, param2=10, minRadius=0, maxRadius=5)
#     dst = gray_sclae.copy()
#     cx, cy, radius = circles[0][0]

#     duration = (time.perf_counter_ns() - start) / BATCH_SIZE
#     avg_duration += duration
#     # print(f"inference time : {duration // 1000000}ms.")
# print(f"avg inference time : {(avg_duration / 1000) // 1000000}ms.")