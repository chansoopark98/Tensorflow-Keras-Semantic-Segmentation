from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import mixed_precision
from models.model_builder import semantic_model
from utils.load_semantic_datasets import SemanticGenerator
from utils.loss import distribute_ce_loss, SparseCategoricalFocalLoss, bce_loss, DistributeLoss
from utils.metrics import MIoU
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.compiler import tensorrt as trt
import numpy as np


class ModelConfiguration():
    def __init__(self, args, mirrored_strategy=None):
        self.args = args
        self.mirrored_strategy = mirrored_strategy

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
        self.NUM_CLASSES = self.args.num_classes
        self.OPTIMIZER_TYPE = self.args.optimizer
        self.BATCH_SIZE = self.args.batch_size
        self.EPOCHS = self.args.epoch
        self.INIT_LR = self.args.lr
        self.SAVE_MODEL_NAME = self.args.model_name + '_' + self.args.model_prefix
        self.DATASET_DIR = self.args.dataset_dir
        self.CHECKPOINT_DIR = self.args.checkpoint_dir
        self.TENSORBOARD_DIR = self.args.tensorboard_dir
        self.IMAGE_SIZE = self.args.image_size
        self.USE_WEIGHT_DECAY = self.args.use_weightDecay
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

            # 2.9.2
            #mixed_precision.set_global_policy('mixed_float16')
            # self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)

    
    def configuration_model(self, image_size=None, num_classes=None):
        if image_size is None:
            self.model = semantic_model(image_size=self.IMAGE_SIZE, num_classes=self.NUM_CLASSES, model='EFFV2S')
        else:
            self.model = semantic_model(image_size=image_size, num_classes=num_classes, model='EFFV2S')
            return self.model

    
    def configuration_metric(self):
        mIoU = MIoU(self.NUM_CLASSES)
        self.metrics = [mIoU]


    def train(self):
        self.__set_args()
        self.__set_callbacks()
        self.configuration_dataset()
        self.__set_optimizer()
        self.configuration_model()
        self.configuration_metric()

        # SparseCategoricalFocalLoss(gamma=2, from_logits=True, use_multi_gpu=self.DISTRIBUTION_MODE)

        
        loss = DistributeLoss(global_batch_size=self.BATCH_SIZE)
        self.model.compile(
            optimizer=self.optimizer,
            loss=loss.ce_loss,
            metrics=self.metrics
            )

        self.model.summary()

        self.model.fit(self.train_data,
                            validation_data=self.valid_data,
                            steps_per_epoch=self.steps_per_epoch,
                            validation_steps=self.validation_steps,
                            epochs=self.EPOCHS,
                            callbacks=self.callback)

        self.model.save_weights(self.CHECKPOINT_DIR + '_' + self.SAVE_MODEL_NAME + '_final_loss.h5')


    def saved_model(self):
        self.__set_args()
        self.configuration_model()
        self.model.load_weights(self.args.saved_model_path)
        export_path = os.path.join(self.CHECKPOINT_DIR, 'export_path', '1')
        os.makedirs(export_path, exist_ok=True)
        self.export_path = export_path
        self.model.summary()
        tf.keras.models.save_model(
            self.model,
            self.export_path,
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None
        )
        print("save model clear")
    
    
    def convert_to_trt(self):
        
        
        # self.model.load_weights(self.args.saved_model_path)
        self.IMAGE_SIZE = (224, 224)
        input_saved_model_dir = './checkpoints/export_path/1/'
        output_saved_model_dir = './checkpoints/export_path_trt/1/'

        os.makedirs(output_saved_model_dir, exist_ok=True)

        params = tf.experimental.tensorrt.ConversionParams(
                                precision_mode='FP16',
                                maximum_cached_engines=16)
        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=input_saved_model_dir, conversion_params=params, use_dynamic_shape=False)
        converter.convert()

        # Define a generator function that yields input data, and use it to execute
        # the graph to build TRT engines.
        def my_input_fn():
            
            inp1 = tf.random.normal((1, 224, 224, 3), dtype=tf.float32)
            inp2 = np.random.normal(size=(1,self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3)).astype(np.float32)
            yield [inp1]
        
        converter.build(input_fn=my_input_fn)  # Generate corresponding TRT engines
        converter.save(output_saved_model_dir)  # Generated engines will be saved.


        