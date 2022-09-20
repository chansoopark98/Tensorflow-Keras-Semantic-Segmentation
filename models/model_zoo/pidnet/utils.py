import tensorflow as tf

import numpy as np


bn_momentum = 0.1

class BasicBlock(tf.keras.layers.Layer):
    
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=stride, padding="same", use_bias=False, 
                                            kernel_initializer='he_uniform')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=1, padding="same", use_bias=False,
                                            kernel_initializer='he_uniform')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def call(self, inputs):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(tf.keras.layers.Layer):
    
    expansion = 2

    def __init__(self, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=planes, kernel_size=1, strides=1, padding="same", use_bias=False, 
                                            kernel_initializer='he_uniform')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        
        self.conv2 = tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=stride, padding="same", use_bias=False,
                                            kernel_initializer='he_uniform')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.conv3 = tf.keras.layers.Conv2D(filters=planes * self.expansion, kernel_size=1, strides=1, padding="same", use_bias=False,
                                            kernel_initializer='glorot_uniform')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.relu = tf.keras.layers.ReLU()

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def call(self, inputs):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class SegmentHead(tf.keras.layers.Layer):
    
    def __init__(self, interplanes, outplanes, scale_factor=None, use_sigmoid=False):
        super(SegmentHead, self).__init__()
        
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

        self.conv1 = tf.keras.layers.Conv2D(filters=interplanes, kernel_size=3, strides=1, padding="same", use_bias=False, 
                                            kernel_initializer='he_uniform')
        
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)       

        self.conv2 = tf.keras.layers.Conv2D(filters=outplanes, kernel_size=1, strides=1, padding="valid", use_bias=True,
                                            kernel_initializer='he_uniform')

        self.relu = tf.keras.layers.ReLU()

        self.scale_factor = scale_factor
        self.use_sigmoid = use_sigmoid
        
        if use_sigmoid:
            self.sigmoid = tf.keras.layers.Activation('sigmoid')


    def call(self, inputs):

        x = self.conv1(self.relu(self.bn1(inputs)))
        out = self.conv2(self.relu(self.bn2(x)))
        
        if self.use_sigmoid:
            out = self.sigmoid(out)

        if self.scale_factor is not None:
            height = tf.shape(x)[1] * self.scale_factor
            width = tf.shape(x)[2] * self.scale_factor

            out = tf.image.resize(out, size=(height, width))
            
        return out


class ScaleProcessBlock(tf.keras.layers.Layer):

    def __init__(self, planes, kernel_size=1, use_pooling=True, pooling_type='average', pool_size=5):
        super(ScaleProcessBlock, self).__init__()

        self.bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(filters=planes, kernel_size=kernel_size, padding='same', use_bias=False, 
                                           kernel_initializer='he_uniform')
        
        self.use_pooling = use_pooling

        if self.use_pooling:
            if pooling_type == 'average':
                self.pool = tf.keras.layers.AveragePooling2D(pool_size=(pool_size, pool_size), 
                                                             strides=(pool_size // 2, pool_size // 2), 
                                                             padding='same')
            elif pooling_type == 'global':
                self.pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)


    def call(self, inputs):

        if self.use_pooling:
            x = self.pool(inputs)
        else:
            x = inputs

        return self.conv(self.relu(self.bn(x)))

        

class DAPPM(tf.keras.layers.Layer):

    def __init__(self, branch_planes, outplanes):
        super(DAPPM, self).__init__()

        self.scales = [
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=False),
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=True, pooling_type='average', pool_size=5),
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=True, pooling_type='average', pool_size=9),
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=True, pooling_type='average', pool_size=17),
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=True, pooling_type='global')
        ]

        self.processes = [
            tf.keras.layers.Layer(),
            ScaleProcessBlock(branch_planes, kernel_size=3, use_pooling=False),
            ScaleProcessBlock(branch_planes, kernel_size=3, use_pooling=False),
            ScaleProcessBlock(branch_planes, kernel_size=3, use_pooling=False),
            ScaleProcessBlock(branch_planes, kernel_size=3, use_pooling=False)
        ]

        self.compression = ScaleProcessBlock(branch_planes * 5, kernel_size=1, use_pooling=False)
        self.shortcut = ScaleProcessBlock(outplanes, kernel_size=1, use_pooling=False)

    def call(self, inputs):
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        xs = [
            self.processes[i](tf.image.resize(self.scales[i](inputs), (height, width)))
            for i in range(5)
        ]

        xs = tf.math.cumsum(xs)
        
        out = self.compression(tf.concat(xs, axis=-1)) + self.shortcut(inputs)

        return out


class PAPPM(tf.keras.layers.Layer):

    def __init__(self, branch_planes, outplanes):
        super(PAPPM, self).__init__()

        self.scales = [
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=False),
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=True, pooling_type='average', pool_size=5),
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=True, pooling_type='average', pool_size=9),
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=True, pooling_type='average', pool_size=17),
            ScaleProcessBlock(branch_planes, kernel_size=1, use_pooling=True, pooling_type='global')
        ]

        self.scale_process = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=branch_planes * 4, kernel_size=3, padding='same', groups=4,
                                   use_bias=False, kernel_initializer='he_uniform')
        ])

        self.compression = ScaleProcessBlock(outplanes, kernel_size=1, use_pooling=False)
        self.shortcut = ScaleProcessBlock(outplanes, kernel_size=1, use_pooling=False)

    def call(self, inputs):
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        xs = [
            tf.image.resize(self.scales[i](inputs), (height, width))
            for i in range(5)
        ]

        xs[1:] = [xs[i] + xs[0] for i in range(1, 5)]
        scale_out = self.scale_process(tf.concat(xs, axis=-1))
        
        out = self.compression(tf.concat([xs[0], scale_out], axis=-1)) + self.shortcut(inputs)

        return out


class PagFM(tf.keras.layers.Layer):

    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False):
        super(PagFM, self).__init__()

        self.with_channel = with_channel
        self.after_relu = after_relu

        self.f_x = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=mid_channels, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        ])

        self.f_y = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=mid_channels, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        ])

        if with_channel:
            self.up = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=in_channels, kernel_size=1, use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ])

        if after_relu:
            self.relu = tf.keras.layers.ReLU()

    
    def call(self, inputs):
        height = tf.shape(inputs[0])[1]
        width = tf.shape(inputs[0])[2]

        if self.after_relu:
            x = self.relu(inputs[0])
            y = self.relu(inputs[1])
        else:
            x, y = inputs

        y_q = self.f_y(y)
        y_q = tf.image.resize(y_q, size=(height, width))

        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = tf.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = tf.sigmoid(tf.reduce_sum(x_k * y_q, axis=-1, keepdims=True))

        y = tf.image.resize(y, size=(height, width))
        x = (1 - sim_map) * x + sim_map * y

        return x


class LightBag(tf.keras.layers.Layer):

    def __init__(self, out_channels):

        super(LightBag, self).__init__()

        self.conv_p = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        ])

        self.conv_i = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        ])

    def call(self, inputs):

        p, i, d = inputs

        edge_att = tf.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class DDFMv2(tf.keras.layers.Layer):

    def __init__(self, out_channels):
        super(DDFMv2, self).__init__()

        self.conv_p = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        ])

        self.conv_i = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        ])

    def call(self, inputs):

        p, i, d = inputs

        edge_att = tf.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class Bag(tf.keras.layers.Layer):

    def __init__(self, out_channels):

        super(Bag, self).__init__()

        self.conv = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding='same', use_bias=False),
        ])

    def call(self, inputs):

        p, i, d = inputs

        egde_att = tf.sigmoid(d)

        return self.conv(edge_att * p + (1 - edge_att) * i)


        