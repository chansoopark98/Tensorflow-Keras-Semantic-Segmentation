import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice


def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


class MIoU(tf.keras.metrics.MeanIoU):
  def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)

        return super().update_state(y_true, y_pred, sample_weight)


class CityMIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.squeeze(y_true, axis=-1)
        y_true += 1
        y_true_mask = tf.where(tf.greater(y_true, 21), 0, 1)
        y_true_mask = tf.cast(y_true_mask, tf.int32)
        y_true = y_true * y_true_mask
        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_pred += 1
        y_pred = tf.cast(y_pred, tf.int32)

        zeros_y_pred = tf.zeros(tf.shape(y_pred), tf.int32)
        zeros_y_pred += y_pred
        indices = tf.cast(tf.where(tf.equal(y_true, 0), 0, 1), tf.int32)

        y_true *= indices
        zeros_y_pred *= indices

        return super().update_state(y_true, zeros_y_pred, sample_weight)


class CityEvalMIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred += 1

        zeros_y_pred = tf.zeros(tf.shape(y_pred), tf.int32)
        zeros_y_pred += y_pred
        indices = tf.cast(tf.where(tf.equal(y_true, 0), 0, 1), tf.int32)

        y_true *= indices
        zeros_y_pred *= indices

        return super().update_state(y_true, zeros_y_pred, sample_weight)