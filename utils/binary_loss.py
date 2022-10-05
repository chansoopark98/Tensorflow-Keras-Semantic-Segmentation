from tensorflow.keras import losses
import tensorflow as tf
import itertools
from typing import Any, Optional

_EPSILON = tf.keras.backend.epsilon()

@tf.keras.utils.register_keras_serializable()
class BinaryBoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits: bool = False, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 1,
                 boundary_alpha: float = 20., **kwargs):
        """
        Args:
            BoundaryLoss is the sum of semantic segmentation loss.
            The BoundaryLoss loss is a binary cross entropy loss.
            
            from_logits       (bool)  : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool)  : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)   : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)   : Number of classes to classify (must be equal to number of last filters in the model)
            boundary_alpha    (float) : Boundary loss alpha
        """
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
        self.boundary_alpha = boundary_alpha
    
        if self.use_multi_gpu:
            self.loss_reduction = losses.Reduction.NONE
        else:
            self.loss_reduction = losses.Reduction.AUTO

    def get_config(self):
        config = super().get_config()
        config.update(from_logits=self.from_logits, use_multi_gpu=self.use_multi_gpu,
                      global_batch_size=self.global_batch_size, num_classes=self.num_classes)
        return config


    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # Calc bce loss                
        edge_map = tf.cast(y_true, dtype=tf.float32)
        grad_components = tf.image.sobel_edges(edge_map)
        grad_mag_components = grad_components ** 2

        grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)

        edge = tf.sqrt(grad_mag_square)
        edge = tf.cast(tf.where(edge>=0.1, 1., 0.), dtype=tf.float32)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=self.from_logits, reduction=self.loss_reduction)(y_true=edge, y_pred=y_pred)

        # Reduce loss to scalar
        if self.use_multi_gpu:
            loss = tf.reduce_mean(loss)
        
        loss *= self.boundary_alpha
        return loss


@tf.keras.utils.register_keras_serializable()
class BinaryAuxiliaryLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits: bool = False, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 1,
                 aux_alpha: float = 0.4,
                  **kwargs):
        """
        Args:
            AuxiliaryLoss is the sum of semantic segmentation loss.
            The AuxiliaryLoss loss is a cross entropy loss.
              
            from_logits       (bool)  : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool)  : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)   : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)   : Number of classes to classify (must be equal to number of last filters in the model)
            aux_alpha         (float) : Aux loss alpha.
        """
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
        self.aux_alpha = aux_alpha
        
        if self.use_multi_gpu:
            self.loss_reduction = losses.Reduction.NONE
        else:
            self.loss_reduction = losses.Reduction.AUTO

    def get_config(self):
        config = super().get_config()
        config.update(from_logits=self.from_logits, use_multi_gpu=self.use_multi_gpu,
                      global_batch_size=self.global_batch_size, num_classes=self.num_classes)
        return config


    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=self.from_logits, reduction=self.loss_reduction)(y_true=y_true, y_pred=y_pred)
        
        if self.use_multi_gpu:
            loss = tf.reduce_mean(loss)

        loss *= self.aux_alpha
        return loss


@tf.keras.utils.register_keras_serializable()
class BinarySegmentationLoss(tf.keras.losses.Loss):
    def __init__(self, gamma, class_weight: Optional[Any] = None,
                 from_logits: bool = False, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 3,
                 dataset_name: str = 'cityscapes',
                 loss_type: str = 'focal',
                  **kwargs):
        """
        Args:
            SemanticLoss is the sum of semantic segmentation loss and confidence loss.
            The semantic loss is a sparse categorical loss,
            and the confidence loss is calculated as binary cross entropy.
              
            gamma             (float): Focal loss's gamma.
            class_weight      (Array): Cross-entropy loss's class weight (logit * class_weight)
            from_logits       (bool) : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool) : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)  : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)  : Number of classes to classify (must be equal to number of last filters in the model)
            dataset_type      (str)  : Train dataset type. For Cityscapes, the process of excluding ignore labels is included.
            loss_type         (str)  : Train loss function type.
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weight = class_weight
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.loss_type = loss_type

        if self.use_multi_gpu:
            self.loss_reduction = losses.Reduction.NONE
        else:
            self.loss_reduction = losses.Reduction.AUTO


    def get_config(self):
        config = super().get_config()
        config.update(gamma=self.gamma, class_weight=self.class_weight,
                      from_logits=self.from_logits, use_multi_gpu=self.use_multi_gpu,
                      global_batch_size=self.global_batch_size, num_classes=self.num_classes)
        return config


    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        
        binary_loss = self.binary_focal_loss(y_true=y_true, y_pred=y_pred)

        return binary_loss


    def binary_focal_loss(self, y_true, y_pred):

        loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=self.from_logits, reduction=self.loss_reduction)(y_true=y_true, y_pred=y_pred)

        if self.use_multi_gpu:
            loss = tf.reduce_mean(loss)

        loss *= 2.

        return loss