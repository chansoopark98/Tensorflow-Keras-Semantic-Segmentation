from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from models.model_builder import base_model
from utils.load_datasets import DatasetGenerator
from utils.loss import bce_loss, dice_loss, total_loss
from utils.metrics import iou_coef, dice_coef
import argparse
import time
import os
import tensorflow as tf


# from utils.cityscape_colormap import class_weight
# from utils.adamW import LearningRateScheduler, poly_decay
# import tensorflow_addons
# sudo apt-get install libtcmalloc-minimal4
# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py
# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py
# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python gan_train.py


tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix",     type=str,   help="Model name", default='L-dice_B-16_E-10_Optim-adam')
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=16)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=100)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--optimizer",     type=str,   help="Optimizer", default='adam')
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard/')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=False)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=True)
parser.add_argument("--distribution_mode",  type=bool,  help="분산 학습 모드 설정", default=True)

args = parser.parse_args()
MODEL_PREFIX = args.model_prefix
WEIGHT_DECAY = args.weight_decay
OPTIMIZER_TYPE = args.optimizer
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name + '_' + args.model_prefix
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
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

TRAIN_INPUT_IMAGE_SIZE = IMAGE_SIZE
VALID_INPUT_IMAGE_SIZE = IMAGE_SIZE
train_dataset_config = DatasetGenerator(DATASET_DIR, TRAIN_INPUT_IMAGE_SIZE, BATCH_SIZE, mode='train')
valid_dataset_config = DatasetGenerator(DATASET_DIR, VALID_INPUT_IMAGE_SIZE, BATCH_SIZE, mode='validation')

train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)
# train_data = mirrored_strategy.experimental_distribute_dataset(train_data)
valid_data = valid_dataset_config.get_validData(valid_dataset_config.valid_data)
# valid_data = mirrored_strategy.experimental_distribute_dataset(valid_data)

steps_per_epoch = train_dataset_config.number_train // BATCH_SIZE
validation_steps = valid_dataset_config.number_valid // BATCH_SIZE

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)

checkpoint_val_loss = ModelCheckpoint(CHECKPOINT_DIR + '_' + SAVE_MODEL_NAME + '_best_loss.h5',
                                      monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
checkpoint_val_iou = ModelCheckpoint(CHECKPOINT_DIR + '_' + SAVE_MODEL_NAME + '_best_iou.h5',
                                      monitor='iou_coef', save_best_only=True, save_weights_only=True,
                                      verbose=1, mode='max')

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR+MODEL_PREFIX, write_graph=True, write_images=True)

polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=base_lr,
                                                          decay_steps=EPOCHS,
                                                          end_learning_rate=0.0001, power=0.9)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay,verbose=1)

if OPTIMIZER_TYPE == 'sgd':
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=base_lr)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
    # optimizer =  tfa.optimizers.RectifiedAdam(learning_rate=base_lr,
    #                                           weight_decay=0.0001,
    #                                           total_steps=int(train_dataset_config.number_train / ( BATCH_SIZE / EPOCHS)),
    #                                           warmup_proportion=0.1,
    #                                           min_lr=0.0001)


if MIXED_PRECISION:
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전


callback = [checkpoint_val_iou, checkpoint_val_loss,  tensorboard, lr_scheduler]

model = base_model(image_size=IMAGE_SIZE)

model.compile(
    optimizer=optimizer,
    loss=bce_loss,
    metrics=[iou_coef, dice_coef]
    )
# tf.keras.losses.BinaryCrossentropy()
if LOAD_WEIGHT:
    weight_name = '_1002_best_miou'
    model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

model.summary()

history = model.fit(train_data,
                    validation_data=valid_data,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=EPOCHS,
                    callbacks=callback)

model.save_weights(CHECKPOINT_DIR + '_' + SAVE_MODEL_NAME + '_final_loss.h5')
#