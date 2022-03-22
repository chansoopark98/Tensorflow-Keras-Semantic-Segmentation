from tensorflow.keras import backend as K
from tensorflow.keras import losses
import tensorflow as tf


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float16)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


def bce_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False)


def total_loss(y_true, y_pred):
    numerator = 2. * K.sum(y_true * y_pred)
    denominator = K.sum(y_true + y_pred)
    dice_loss = K.mean(1 - numerator / denominator)
    bce_loss = losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False)
    return dice_loss + bce_loss

    






