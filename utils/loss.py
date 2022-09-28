from os import posix_fadvise
from tensorflow.keras import losses
import tensorflow as tf
import itertools
from typing import Any, Optional

_EPSILON = tf.keras.backend.epsilon()


@tf.keras.utils.register_keras_serializable()
class BoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits: bool = False, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 1,
                  **kwargs):
        """
        Args:
            BoundaryLoss is the sum of semantic segmentation loss.
            The BoundaryLoss loss is a binary cross entropy loss.
            
            from_logits       (bool) : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool) : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)  : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)  : Number of classes to classify (must be equal to number of last filters in the model)
        """
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
    
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
        edge = tf.where(edge != 0, 1, 0)

        loss = losses.BinaryCrossentropy(from_logits=True, reduction=self.loss_reduction)(y_true=edge, y_pred=y_pred)

        # Reduce loss to scalar
        if self.use_multi_gpu:
            loss = tf.reduce_mean(loss)

        return loss



@tf.keras.utils.register_keras_serializable()
class AuxiliaryLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits: bool = False, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 1,
                  **kwargs):
        """
        Args:
            AuxiliaryLoss is the sum of semantic segmentation loss.
            The AuxiliaryLoss loss is a cross entropy loss.
              
            from_logits       (bool) : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool) : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)  : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)  : Number of classes to classify (must be equal to number of last filters in the model)
        """
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
    
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
        loss = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=self.loss_reduction)(y_true=y_true, y_pred=y_pred)
             
        if self.use_multi_gpu:
            loss = tf.reduce_mean(loss)
        
        return loss





@tf.keras.utils.register_keras_serializable()
class SemanticLoss(tf.keras.losses.Loss):
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
        if self.dataset_name == 'cityscapes':
            # ignore label indexing
            y_true = tf.where(y_true==-1, 255, y_true)

            y_true = tf.squeeze(y_true, axis=3)
            y_true = tf.reshape(y_true, [-1,])
            
            y_pred = tf.reshape(y_pred, [-1, self.num_classes])
            indices = tf.squeeze(tf.where(tf.less_equal(y_true, self.num_classes-1)), 1)

            # gather by indices(valid labels)
            y_true = tf.cast(tf.gather(y_true, indices), tf.int32)
            y_pred = tf.gather(y_pred, indices)

        
        if self.loss_type == 'focal':
            semantic_loss = self.sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred,
                                                class_weight=self.class_weight,
                                                gamma=self.gamma,
                                                from_logits=self.from_logits,
                                                use_multi_gpu=self.use_multi_gpu,
                                                global_batch_size=self.global_batch_size)
        elif self.loss_type == 'ce':
            semantic_loss = self.sparse_categorical_cross_entropy(y_true=y_true,
                                                                  y_pred=y_pred,
                                                                  use_multi_gpu=self.use_multi_gpu)

        return semantic_loss


    def sparse_categorical_cross_entropy(self, y_true, y_pred, use_multi_gpu):
        ce_loss = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=self.loss_reduction)(y_true=y_true, y_pred=y_pred)
             
        if use_multi_gpu:
            ce_loss = tf.reduce_mean(ce_loss)
        
        return ce_loss
        
    
    def sparse_categorical_focal_loss(self, y_true, y_pred, gamma, *,
                                  class_weight: Optional[Any] = None,
                                  from_logits: bool = False, axis: int = -1,
                                  use_multi_gpu: bool = False,
                                  global_batch_size: int = 8,
                                  ) -> tf.Tensor:
        # Process focusing parameter
        gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
        gamma_rank = gamma.shape.rank
        scalar_gamma = gamma_rank == 0

        # Process class weight
        if class_weight is not None:
            class_weight = tf.convert_to_tensor(class_weight,
                                                dtype=tf.dtypes.float32)

        # Process prediction tensor
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred_rank = y_pred.shape.rank
        if y_pred_rank is not None:
            axis %= y_pred_rank
            if axis != y_pred_rank - 1:
                # Put channel axis last for sparse_softmax_cross_entropy_with_logits
                perm = list(itertools.chain(range(axis),
                                            range(axis + 1, y_pred_rank), [axis]))
                y_pred = tf.transpose(y_pred, perm=perm)
        elif axis != -1:
            raise ValueError(
                f'Cannot compute sparse categorical focal loss with axis={axis} on '
                'a prediction tensor with statically unknown rank.')
        y_pred_shape = tf.shape(y_pred)

        # Process ground truth tensor
        y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
        y_true_rank = y_true.shape.rank

        if y_true_rank is None:
            raise NotImplementedError('Sparse categorical focal loss not supported '
                                    'for target/label tensors of unknown rank')

        reshape_needed = (y_true_rank is not None and y_pred_rank is not None and
                        y_pred_rank != y_true_rank + 1)
        if reshape_needed:
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

        if from_logits:
            logits = y_pred
            probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            probs = y_pred
            logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

        xent_loss = losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=self.loss_reduction)(y_true=y_true, y_pred=logits)

        if use_multi_gpu:
            xent_loss = tf.reduce_mean(xent_loss)
        

        y_true_rank = y_true.shape.rank
        probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)

        if not scalar_gamma:
            gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)
        focal_modulation = (1 - probs) ** gamma

        loss = focal_modulation * xent_loss



        if class_weight is not None:
            class_weight = tf.gather(class_weight, y_true, axis=0,
                                    batch_dims=y_true_rank)
            loss *= class_weight

        if reshape_needed:
            loss = tf.reshape(loss, y_pred_shape[:-1])

        return loss