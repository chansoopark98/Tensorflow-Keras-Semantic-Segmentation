from tensorflow.keras.layers import (AveragePooling2D,
    MaxPooling2D, SeparableConv2D, UpSampling2D, Activation, BatchNormalization,
    GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, multiply, Add, concatenate,
    DepthwiseConv2D, Reshape, ZeroPadding2D, Dense, GlobalMaxPooling2D, Permute, Lambda, Subtract)
import tensorflow as tf

MOMENTUM = 0.99
EPSILON = 1e-3
DECAY = None
BN = BatchNormalization
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_out", distribution="truncated_normal")
atrous_rates= (6, 12, 18)

def deepLabV3Plus(features, base_channel=256, activation='swish'):
    skip1, x = features # c1 48 / c2 64

    # Image Feature branch
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(base_channel, (1, 1), padding='same',
                kernel_regularizer=DECAY,
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=EPSILON)(b4)
    b4 = Activation(activation)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)
    # b4 = UpSampling2D((14, 14), interpolation="bilinear")(b4)

    # b4 = UpSampling2D(size=(32, 64), interpolation="bilinear")(b4)
    # simple 1x1
    b0 = Conv2D(base_channel, (1, 1), padding='same',
                kernel_regularizer=DECAY,
                use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=EPSILON)(b0)
    b0 = BN(name='aspp0_BN', epsilon=EPSILON)(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    b1 = SepConv_BN(x, base_channel, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=EPSILON)
    # rate = 12 (24)
    b2 = SepConv_BN(x, base_channel, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=EPSILON)
    # rate = 18 (36)
    b3 = SepConv_BN(x, base_channel, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=EPSILON)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(base_channel, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=EPSILON)(x)
    x = BN(name='concat_projection_BN', epsilon=EPSILON)(x)
    x = Activation(activation)(x)

    x = Dropout(0.1)(x)

    size_before = tf.keras.backend.int_shape(skip1)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(x)
    # x = UpSampling2D((4, 4), interpolation="bilinear")(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       kernel_regularizer=DECAY,
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BN(
        name='feature_projection0_BN', epsilon=EPSILON)(dec_skip1)
    dec_skip1 = Activation(activation)(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, base_channel, 'decoder_conv0',
                   depth_activation=True, epsilon=EPSILON)
    x = SepConv_BN(x, base_channel, 'decoder_conv1',
                   depth_activation=True, epsilon=EPSILON)


    return x

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    activation = 'swish'
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(activation)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        kernel_regularizer=DECAY,
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    # x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    x = BN(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               kernel_regularizer=DECAY,
               use_bias=False, name=prefix + '_pointwise')(x)
    # x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    x = BN(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)

    return x