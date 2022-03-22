from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from models.model_builder import base_model
from utils.load_datasets import DatasetGenerator
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# from utils.cityscape_colormap import class_weight
# from utils.adamW import LearningRateScheduler, poly_decay
# import tensorflow_addons
# sudo apt-get install libtcmalloc-minimal4
# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py
# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py
# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python gan_train.py


tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
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


TRAIN_INPUT_IMAGE_SIZE = IMAGE_SIZE
VALID_INPUT_IMAGE_SIZE = IMAGE_SIZE
test_dataset_config = DatasetGenerator(DATASET_DIR, TRAIN_INPUT_IMAGE_SIZE, BATCH_SIZE, mode='validation')
# valid_dataset_config = DatasetGenerator(DATASET_DIR, VALID_INPUT_IMAGE_SIZE, BATCH_SIZE, mode='validation', model_name='effnet')

test_set = test_dataset_config.get_testData(test_dataset_config.valid_data)
# train_data = mirrored_strategy.experimental_distribute_dataset(train_data)
# valid_data = valid_dataset_config.get_validData(valid_dataset_config.valid_data)
# valid_data = mirrored_strategy.experimental_distribute_dataset(valid_data)
#
test_steps = test_dataset_config.number_valid // BATCH_SIZE


model = base_model(image_size=IMAGE_SIZE)


weight_name = '_0322_best_loss'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

model.summary()
batch_idx = 0
for x, y, original in tqdm(test_set, total=test_steps):
    pred = model.predict_on_batch(x)

    img = x[0]
    pred = pred[0]
    label = y[0]
    original = original[0]

    pred = tf.where(pred== 1.0, 1, 0)

    original = tf.cast(original, tf.int32)

    output = original * pred

    rows = 1
    cols = 4
    fig = plt.figure()

    
    img = tf.cast(img, tf.float32)

    ax0 = fig.add_subplot(rows, cols, 1)
    ax0.imshow(original)
    ax0.set_title('img')
    ax0.axis("off")

    ax0 = fig.add_subplot(rows, cols, 2)
    ax0.imshow(label) 
    ax0.set_title('label')
    ax0.axis("off")

    ax1 = fig.add_subplot(rows, cols, 3)
    ax1.imshow(pred)
    ax1.set_title('pred')
    ax1.axis("off")

    ax1 = fig.add_subplot(rows, cols, 4)
    ax1.imshow(output)
    ax1.set_title('pred')
    ax1.axis("off")

    batch_idx += 1
    tf.keras.preprocessing.image.save_img(MASK_RESULT_DIR + str(batch_idx) + 'output_mask.png', output)
    plt.savefig(RESULT_DIR + str(batch_idx) + '_output.png', dpi=300)








