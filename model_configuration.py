from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
from models.model_zoo.pidnet.pidnet import PIDNet
from models.model_builder import ModelBuilder
from utils.load_semantic_datasets import SemanticGenerator
from utils.loss import SemanticLoss
from utils.metrics import MIoU, CityMIoU
import os
import tensorflow as tf
import tensorflow_addons as tfa


class ModelConfiguration(SemanticGenerator):
    def __init__(self, args: object, mirrored_strategy: object = None):
        """
        Args:
            args (argparse): Training options (argparse).
            mirrored_strategy (tf.distribute): tf.distribute.MirroredStrategy() with Session.
        """
        self.args = args
        self.mirrored_strategy = mirrored_strategy
        self.__set_args()
        super().__init__(data_dir=self.DATASET_DIR, image_size=self.IMAGE_SIZE,
                         batch_size=self.BATCH_SIZE, dataset_name=self.DATASET_NAME)


    def configuration_dataset(self):
        """
            Configure the dataset. Train and validation dataset is inherited from the parent class and used.
        """
        # Wrapping tf.data generator
        self.train_data = self.get_trainData(train_data=self.train_data)
        self.valid_data = self.get_validData(valid_data=self.valid_data)
    
        # Calculate training and validation steps
        self.steps_per_epoch = self.number_train // self.BATCH_SIZE
        self.validation_steps = self.number_valid // self.BATCH_SIZE

        # Wrapping tf.data generator if when use multi-gpu training
        if self.DISTRIBUTION_MODE:
            self.train_data = self.mirrored_strategy.experimental_distribute_dataset(self.train_data)
            self.valid_data = self.mirrored_strategy.experimental_distribute_dataset(self.valid_data)   


    def __set_args(self):
        """
            Configure the options received from argparse.
        """
        # Set training variables from argparse's arguments 
        self.MODEL_PREFIX = self.args.model_prefix
        self.WEIGHT_DECAY = self.args.weight_decay
        self.NUM_CLASSES = self.args.num_classes
        self.OPTIMIZER_TYPE = self.args.optimizer
        self.BATCH_SIZE = self.args.batch_size
        self.EPOCHS = self.args.epoch
        self.INIT_LR = self.args.lr
        self.SAVE_MODEL_NAME = self.args.model_name + '_' + self.args.model_prefix
        self.DATASET_DIR = self.args.dataset_dir
        self.DATASET_NAME = self.args.dataset_name
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
        """
            Set the keras callback of model.fit.

            For some metric callbacks, the name of the custom metric may be different and may not be valid,
            so you must specify the name of the custom metric.
        """
        # Set training keras callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)
        
        checkpoint_val_loss = ModelCheckpoint(self.CHECKPOINT_DIR + self.args.model_name + '/_' + self.SAVE_MODEL_NAME + '_best_loss.h5',
                                              monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
        
        checkpoint_val_iou = ModelCheckpoint(self.CHECKPOINT_DIR + self.args.model_name + '/_' + self.SAVE_MODEL_NAME + '_best_iou.h5',
                                             monitor=self.miou_name, save_best_only=True, save_weights_only=True,
                                             verbose=1, mode='max')

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=self.TENSORBOARD_DIR + 'semantic/' + self.MODEL_PREFIX, write_graph=True, write_images=True)

        # lrDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=self.INIT_LR,
        #                                                           decay_steps=self.EPOCHS,
        #                                                           end_learning_rate=self.INIT_LR * 0.01, power=0.9)
        
        lrDecay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=self.INIT_LR, decay_steps=self.EPOCHS, alpha=self.INIT_LR * 0.001)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lrDecay, verbose=1)
        
        # If you wanna need another callbacks, please add here.
        self.callback = [checkpoint_val_iou,
                         checkpoint_val_loss,  tensorboard, lr_scheduler]


    def __set_optimizer(self):
        """
            Configure the optimizer for backpropagation calculations.
        """
        if self.OPTIMIZER_TYPE == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=self.INIT_LR)
            self.optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=self.INIT_LR, momentum=0.9)
        elif self.OPTIMIZER_TYPE == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.INIT_LR)
        elif self.OPTIMIZER_TYPE == 'radam':
            self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.INIT_LR,
                                                          weight_decay=0.00001,
                                                          total_steps=int(
                                                          self.number_train / (self.BATCH_SIZE / self.EPOCHS)),
                                                          warmup_proportion=0.1,
                                                          min_lr=0.0001)
            
        if self.MIXED_PRECISION:
            # Wrapping optimizer when use distribute training (multi-gpu training)
            # mixed_precision.set_global_policy('mixed_float16')

            policy = mixed_precision.Policy('mixed_float16')
            
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)

    
    def __configuration_model(self):
        """
            Build a deep learning model.
        """
        from models.model_zoo.PIDNet import PIDNet
        self.model = PIDNet(input_shape=(*self.IMAGE_SIZE, 3), m=2, n=3, num_classes=self.NUM_CLASSES,
                       planes=32, ppm_planes=96, head_planes=128, augment=False, training=True).build()


        
        # self.model = PIDNet(input_shape=(*self.IMAGE_SIZE, 3), m=2, n=3, num_classes=self.NUM_CLASSES,
        #                planes=32, ppm_planes=96, head_planes=128, augment=False)
        # self.model.build((None, *self.IMAGE_SIZE, 3))


        # self.model = ModelBuilder(image_size=self.IMAGE_SIZE,
        #                           num_classes=self.NUM_CLASSES).build_model()



    def __configuration_metric(self):
        """
            Configure metrics for use in training and evaluation.
        """
        if self.DATASET_NAME == 'cityscapes':
            print('cityscapes dataset miou')
            mIoU = CityMIoU(self.NUM_CLASSES+1)
            self.miou_name = 'city_m_io_u'
            
        else:
            print('custom dataset miou')
            mIoU = MIoU(self.NUM_CLASSES)
            self.miou_name = 'm_io_u'
        
        # You can add here custom metrics.
        self.metrics = [mIoU]


    def train(self):
        """
            Compile all configuration settings required for training.
            If the custom metric name is different in the __set_callbacks function,
            the update may not be possible, so please check the name.
        """
        self.configuration_dataset()
        self.__set_optimizer()
        self.__configuration_model()
        self.__configuration_metric()
        self.__set_callbacks()

        self.model.compile(
            optimizer=self.optimizer,
            loss=SemanticLoss(gamma=1.5, from_logits=True, use_multi_gpu=self.DISTRIBUTION_MODE,
                              global_batch_size=self.BATCH_SIZE, num_classes=self.NUM_CLASSES,
                              dataset_name=self.DATASET_NAME),
            metrics=self.metrics)
        # self.model.summary()
        self.model.fit(self.train_data,
                       validation_data=self.valid_data,
                       steps_per_epoch=self.steps_per_epoch,
                       validation_steps=self.validation_steps,
                       epochs=self.EPOCHS,
                       callbacks=self.callback)


    def saved_model(self):
        """
            Convert it to a graph model (.pb) using the learned weights.
        """

        # self.model = PIDNet(input_shape=(*self.IMAGE_SIZE, 3), m=2, n=3, num_classes=self.NUM_CLASSES,
        #                planes=32, ppm_planes=96, head_planes=128, augment=False)
        # self.model.build((None, *self.IMAGE_SIZE, 3))
        # input_arr = tf.random.uniform((1, *self.IMAGE_SIZE, 3))
        # outputs = self.model(input_arr)

        from models.model_zoo.PIDNet import PIDNet
        self.model = PIDNet(input_shape=(*self.IMAGE_SIZE, 3), m=2, n=3, num_classes=self.NUM_CLASSES,
                       planes=32, ppm_planes=96, head_planes=128, augment=False, training=False).build()
        
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