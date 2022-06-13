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
import tensorflow_model_optimization as tfmot
import time
import numpy as np
import tempfile

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix",     type=str,   help="저장된 모델 경로", default='_0613_224_224_mobilenetv3s_test(no_pretrained)')
parser.add_argument("--checkpoint_dir",     type=str,   help="저장된 모델 경로", default='./checkpoints/0613/_0613_224_224_mobilenetv3s_test(no_pretrained)_best_iou.h5')
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=16)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=10)
parser.add_argument("--init_lr",          type=float,   help="에폭 설정", default=0.0001)
parser.add_argument("--optimizer",     type=str,   help="Optimizer", default='adam')
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=False)
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard/')
args = parser.parse_args()


class ConvertToLite():
    def __init__(self, args):
        self.args = args
        self.__set_args()
        self.configuration_dataset()
        self.configuration_model()
        self.__set_optimizer()
        self.__set_callbacks()
        self.configuration_metric()
        

    def configuration_dataset(self):
        self.train_dataset_config = SemanticGenerator(self.DATASET_DIR, self.IMAGE_SIZE, self.BATCH_SIZE, mode='train')
        self.valid_dataset_config = SemanticGenerator(self.DATASET_DIR, self.IMAGE_SIZE, self.BATCH_SIZE, mode='validation')

        self.train_data = self.train_dataset_config.get_trainData(self.train_dataset_config.train_data)
        self.valid_data = self.valid_dataset_config.get_validData(self.valid_dataset_config.valid_data)

        self.steps_per_epoch = self.train_dataset_config.number_train // self.BATCH_SIZE
        self.validation_steps = self.valid_dataset_config.number_valid // self.BATCH_SIZE 


    def __set_args(self):
        self.CHECKPOINT_DIR = args.checkpoint_dir
        self.BATCH_SIZE = args.batch_size
        self.EPOCH = args.epoch
        self.DATASET_DIR = args.dataset_dir
        self.TENSORBOARD_DIR = args.tensorboard_dir
        self.MODEL_PREFIX = args.model_prefix
        self.SAVE_MODEL_NAME = self.args.model_name + '_' + self.args.model_prefix
        self.MIXED_PRECISION = args.mixed_precision
        self.INIT_LR = args.init_lr
        self.OPTIMIZER_TYPE = args.optimizer
        self.IMAGE_SIZE = (224, 224)


    def configuration_model(self):
        self.model = semantic_model(image_size=self.IMAGE_SIZE)


    def __set_optimizer(self):
        if self.OPTIMIZER_TYPE == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=self.INIT_LR)
        elif self.OPTIMIZER_TYPE == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.INIT_LR)
        elif self.OPTIMIZER_TYPE == 'radam':
            self.optimizer =  tfa.optimizers.RectifiedAdam(learning_rate=self.INIT_LR,
                                                    weight_decay=0.00001,
                                                    total_steps=int(self.train_dataset_config.number_train / (self.BATCH_SIZE / self.EPOCH)),
                                                    warmup_proportion=0.1,
                                                    min_lr=0.0001)

        if self.MIXED_PRECISION:
            policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
            mixed_precision.set_policy(policy)
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, loss_scale='dynamic')


    def __set_callbacks(self):
        logdir = tempfile.mkdtemp()

        update_pruning_step = tfmot.sparsity.keras.UpdatePruningStep(),
        pruning_summary = tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)

        checkpoint_val_iou = ModelCheckpoint(self.CHECKPOINT_DIR + self.args.model_name +'/_' + self.SAVE_MODEL_NAME + '_best_iou.h5',
                                            monitor='val_m_io_u', save_best_only=True, save_weights_only=True,
                                            verbose=1, mode='max')

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.TENSORBOARD_DIR +'pruning/' + self.MODEL_PREFIX, write_graph=True, write_images=True)


        self.callback = [update_pruning_step, pruning_summary]


    def configuration_metric(self):
        mIoU = MIoU(2)
        self.metrics = [mIoU]


    def train(self):
        
        self.model.load_weights(self.CHECKPOINT_DIR)
        self.model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(self.model)

        # num_images = self.train_dataset_config.number_train
        # end_step = np.ceil(num_images / self.BATCH_SIZE).astype(np.int32) * self.EPOCH
        # # Define model for pruning.
        # pruning_params = {
        # 'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
        #                                                        final_sparsity=0.80,
        #                                                        begin_step=0,
        #                                                        end_step=end_step)
        # }

        
        # self.model = prune_low_magnitude(self.model, **pruning_params)

        self.model_for_pruning.compile(
            optimizer=self.optimizer,
            loss=SparseCategoricalFocalLoss(gamma=2, from_logits=True),
            metrics=self.metrics
            )


        self.model_for_pruning.summary()

        self.model_for_pruning.fit(self.train_data,
                            validation_data=self.valid_data,
                            steps_per_epoch=self.steps_per_epoch,
                            validation_steps=self.validation_steps,
                            epochs=self.EPOCH,
                            callbacks=self.callback)

        _, pretrained_weights = tempfile.mkstemp('.tf')

        self.model_for_pruning.save_weights(pretrained_weights)

                

if __name__ == '__main__':
    tf.config.set_soft_device_placement(True)

    with tf.device('/device:GPU:0'):
        training = ConvertToLite(args=args)
        training.train()
