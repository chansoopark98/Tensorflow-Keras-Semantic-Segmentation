from tensorflow.keras import layers
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU, MaxPooling2D, Input, Flatten, Dense, Dropout, concatenate,
    DepthwiseConv2D,  ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D)
from tensorflow.keras.activations import tanh, relu
import tensorflow as tf
from tensorflow.keras.models import Model
MOMENTUM = 0.9
EPSILON = 1e-5
# DECAY = tf.keras.regularizers.L2(l2=0.0001/2)
DECAY = None
# BN = tf.keras.layers.experimental.SyncBatchNormalization
BN = BatchNormalization
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = "he_normal"
activation = 'swish'

def conv_module(x, channels, kernel_size=3, strides=1,
                bn_momentum=0.8, activation='swish', dropout=0.4, prefix='name'):
    x = Conv2D(channels,
                     kernel_size=(kernel_size, kernel_size),
                     strides=(strides, strides),
                     use_bias=False,
                     kernel_initializer='he_normal',
                     padding='same',
                     name=prefix+'_conv2d')(x)
    x = BatchNormalization(momentum=bn_momentum, name=prefix+'_bn')(x)

    if dropout != 0.:
        x = Dropout(dropout, name=prefix+'dropout')(x)

    if activation == 'leakyrelu':
        x = Activation(LeakyReLU(0.2), name=prefix+'_activation')(x)
    else:
        x = Activation(activation, name=prefix+'_activation')(x)
    return x

def deconv_module(x, channels, kernel_size=3, strides=2, activation='swish', bn_momentum=0.99, prefix='name', padding='same'):
    x = Conv2DTranspose(channels,
                      kernel_size=(kernel_size, kernel_size),
                      strides=(strides, strides),
                      use_bias=False,
                      kernel_initializer='he_normal',
                      name=prefix+'_upsampling',
                      padding=padding)(x)
    x = BatchNormalization(momentum=bn_momentum, name=prefix+'_bn')(x)
    if activation == 'leakyrelu':
        x = Activation(LeakyReLU(0.2), name=prefix+'_activation')(x)
    else:
        x = Activation(activation, name=prefix+'_activation')(x)
    return x


def unet(input_shape=(300, 300, 3), base_channel=8, output_channel=1, use_logits=False):
    inputs = Input(shape=input_shape, name='image_input')
    bn_momentum = 0.99
    activation = 'relu'
    # Encoder branch
    conv1_1 = conv_module(x=inputs, channels=base_channel, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv1_1')
    conv1_2 = conv_module(x=conv1_1, channels=base_channel, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv1_2')
    conv1_pool = MaxPooling2D((2,2))(conv1_2) # 128x128@64

    conv2_1 = conv_module(x=conv1_pool, channels=base_channel * 2, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv2_1')
    conv2_2 = conv_module(x=conv2_1, channels=base_channel * 2, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv2_2')
    conv2_pool = MaxPooling2D((2, 2))(conv2_2)  # 64x64 @128

    conv3_1 = conv_module(x=conv2_pool, channels=base_channel * 4, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv3_1')
    conv3_2 = conv_module(x=conv3_1, channels=base_channel * 4, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv3_2')
    conv3_pool = MaxPooling2D((2, 2))(conv3_2) # 32x32 @256

    conv4_1 = conv_module(x=conv3_pool, channels=base_channel * 8, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv4_1')
    conv4_2 = conv_module(x=conv4_1, channels=base_channel * 8, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv4_2')
    conv4_pool = MaxPooling2D((2, 2))(conv4_2) # 16x16 @512

    conv5_1 = conv_module(x=conv4_pool, channels=base_channel * 16, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv5_1')
    conv5_2 = conv_module(x=conv5_1, channels=base_channel * 16, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='conv5_2')

    deconv_5 = deconv_module(conv5_2, channels=base_channel * 8, kernel_size=3, strides=2, prefix='deconv_os16')
    
    decoder = Concatenate()([conv4_2, deconv_5])
    decoder = Dropout(0.4)(decoder)

    decoder = conv_module(x=decoder, channels=base_channel * 8, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='decoder_4_1')
    decoder = conv_module(x=decoder, channels=base_channel * 8, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='decoder_4_2')

    decoder = deconv_module(decoder, channels=base_channel * 4, kernel_size=3, strides=2, prefix='deconv_os8')
    decoder = Concatenate()([conv3_2, decoder])
    decoder = Dropout(0.4)(decoder)

    decoder = conv_module(x=decoder, channels=base_channel * 4, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='decoder_3_1')
    decoder = conv_module(x=decoder, channels=base_channel * 4, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                activation=activation, dropout=0.0, prefix='decoder_3_2')

    decoder = deconv_module(decoder, channels=base_channel * 2, kernel_size=3, strides=2, prefix='deconv_os4')
    decoder = Concatenate()([conv2_2, decoder])
    decoder = Dropout(0.4)(decoder)

    decoder = conv_module(x=decoder, channels=base_channel * 2, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                          activation=activation, dropout=0.0, prefix='decoder_2_1')
    decoder = conv_module(x=decoder, channels=base_channel * 2, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                          activation=activation, dropout=0.0, prefix='decoder_2_2')

    decoder = deconv_module(decoder, channels=base_channel, kernel_size=3, strides=2, prefix='deconv_os2')
    decoder = Concatenate()([conv1_2, decoder])
    decoder = Dropout(0.4)(decoder)

    decoder = conv_module(x=decoder, channels=base_channel, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                          activation=activation, dropout=0.0, prefix='decoder_1_1')
    decoder = conv_module(x=decoder, channels=base_channel, kernel_size=3, strides=1, bn_momentum=bn_momentum,
                          activation=activation, dropout=0.0, prefix='decoder_1_2')

    if use_logits:
        outputs = Conv2D(output_channel,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    use_bias=True,
                    kernel_initializer='he_normal',
                    padding='same')(decoder)
    else:
        outputs = Conv2D(output_channel,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    use_bias=True,
                    kernel_initializer='he_normal',
                    padding='same',
                    activation='sigmoid')(decoder)

    return inputs, outputs