from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from models.model_builder import semantic_model
from utils.load_semantic_datasets import SemanticGenerator
from utils.loss import ce_loss, SparseCategoricalFocalLoss
from utils.metrics import MIoU
import argparse
import time
import os
import tensorflow as tf
import tensorflow_addons as tfa

# 1. sudo apt-get install libtcmalloc-minimal4
# 2. check dir ! 
# dpkg -L libtcmalloc-minimal4
# 3. LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
# Model name : ImageSize_BATCH_EPOCH_InitLR_Optimizer_GPU(single or multi)
parser.add_argument("--model_prefix",     type=str,   help="Model name", default='224_8_100_0.001_adam_multi')
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=8)
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
parser.add_argument("--multi_gpu",  help="분산 학습 모드 설정", action='store_true')

args = parser.parse_args()


class Train():
    def __init__(self, args, mirrored_strategy=None):
        self.args = args
        self.mirrored_strategy = mirrored_strategy
        self.__set_args()
        self.__set_callbacks()
        self.configuration_dataset()
        self.__set_optimizer()
        self.configuration_model()
        self.configuration_metric()
        

    def configuration_dataset(self):
        self.train_dataset_config = SemanticGenerator(self.DATASET_DIR, self.IMAGE_SIZE, self.BATCH_SIZE, mode='train')
        self.valid_dataset_config = SemanticGenerator(self.DATASET_DIR, self.IMAGE_SIZE, self.BATCH_SIZE, mode='validation')

        self.train_data = self.train_dataset_config.get_trainData(self.train_dataset_config.train_data)
        self.valid_data = self.valid_dataset_config.get_validData(self.valid_dataset_config.valid_data)

        self.steps_per_epoch = self.train_dataset_config.number_train // self.BATCH_SIZE
        self.validation_steps = self.valid_dataset_config.number_valid // self.BATCH_SIZE

        if self.DISTRIBUTION_MODE:
            self.train_data = self.mirrored_strategy.experimental_distribute_dataset(self.train_data)
            self.valid_data = self.mirrored_strategy.experimental_distribute_dataset(self.valid_data)   


    def __set_args(self):
        self.MODEL_PREFIX = self.args.model_prefix
        self.WEIGHT_DECAY = self.args.weight_decay
        self.OPTIMIZER_TYPE = self.args.optimizer
        self.BATCH_SIZE = self.args.batch_size
        self.EPOCHS = self.args.epoch
        self.INIT_LR = self.args.lr
        self.SAVE_MODEL_NAME = self.args.model_name + '_' + self.args.model_prefix
        self.DATASET_DIR = self.args.dataset_dir
        self.CHECKPOINT_DIR = self.args.checkpoint_dir
        self.TENSORBOARD_DIR = self.args.tensorboard_dir
        self.IMAGE_SIZE = (224, 224)
        self.USE_WEIGHT_DECAY = self.args.use_weightDecay
        self.LOAD_WEIGHT = self.args.load_weight
        self.MIXED_PRECISION = self.args.mixed_precision
        self.DISTRIBUTION_MODE = self.args.multi_gpu
        if self.DISTRIBUTION_MODE:
            self.BATCH_SIZE *= 2

        os.makedirs(self.DATASET_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR + self.args.model_name, exist_ok=True)


    def __set_callbacks(self):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)

        checkpoint_val_loss = ModelCheckpoint(self.CHECKPOINT_DIR + self.args.model_name+ '/_' + self.SAVE_MODEL_NAME + '_best_loss.h5',
                                            monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
        checkpoint_val_iou = ModelCheckpoint(self.CHECKPOINT_DIR + self.args.model_name +'/_' + self.SAVE_MODEL_NAME + '_best_iou.h5',
                                            monitor='val_m_io_u', save_best_only=True, save_weights_only=True,
                                            verbose=1, mode='max')

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.TENSORBOARD_DIR +'semantic/' + self.MODEL_PREFIX, write_graph=True, write_images=True)

        polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=self.INIT_LR,
                                                                decay_steps=self.EPOCHS,
                                                                end_learning_rate=self.INIT_LR * 0.1, power=0.9)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay,verbose=1)

        self.callback = [checkpoint_val_iou, checkpoint_val_loss,  tensorboard, lr_scheduler]


    def __set_optimizer(self):
        if self.OPTIMIZER_TYPE == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=self.INIT_LR)
        elif self.OPTIMIZER_TYPE == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.INIT_LR)
        elif self.OPTIMIZER_TYPE == 'radam':
            self.optimizer =  tfa.optimizers.RectifiedAdam(learning_rate=self.INIT_LR,
                                                    weight_decay=0.00001,
                                                    total_steps=int(self.train_dataset_config.number_train / (self.BATCH_SIZE / self.EPOCHS)),
                                                    warmup_proportion=0.1,
                                                    min_lr=0.0001)

        if self.MIXED_PRECISION:
            policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
            mixed_precision.set_policy(policy)
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, loss_scale='dynamic')

    
    def configuration_model(self):
        self.model = semantic_model(image_size=self.IMAGE_SIZE)

    
    def configuration_metric(self):
        mIoU = MIoU(2)
        self.metrics = [mIoU]


    def train(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=SparseCategoricalFocalLoss(gamma=2, from_logits=True),
            metrics=self.metrics
            )

        if self.LOAD_WEIGHT:
            weight_name = '_1002_best_miou'
            self.model.load_weights(self.CHECKPOINT_DIR + weight_name + '.h5')

        self.model.summary()

        self.model.fit(self.train_data,
                            validation_data=self.valid_data,
                            steps_per_epoch=self.steps_per_epoch,
                            validation_steps=self.validation_steps,
                            epochs=self.EPOCHS,
                            callbacks=self.callback)

        self.model.save_weights(self.CHECKPOINT_DIR + '_' + self.SAVE_MODEL_NAME + '_final_loss.h5')

                

if __name__ == '__main__':
    if args.multi_gpu == False:
        tf.config.set_soft_device_placement(True)

        with tf.device('/device:GPU:0'):
            training = Train(args=args)
            training.train()

    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            training = Train(args=args, mirrored_strategy=mirrored_strategy)
            training.train()
