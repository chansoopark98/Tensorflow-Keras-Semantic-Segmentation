import tensorflow as tf


bn_momentum = 0.1


class PIDNet(tf.keras.models.Model):

    def __init__(self, input_shape, m=2, n=3, num_classes=11, planes=64, ppm_planes=96, head_planes=128, augment=True, use_sigmoid=False):
        super(PIDNet, self).__init__()

        self.augment = augment

        # I-branch

        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
            tf.keras.layers.ReLU()
        ], name='pid_net_conv1')

        self.relu = tf.keras.layers.ReLU(name='pidnet_relu')

        self.layer1_i = self._make_layer(BasicBlock, planes, planes, num_blocks=m, stride=1)
        self.layer2_i = self._make_layer(BasicBlock, planes, planes * 2, num_blocks=m, stride=2)
        self.layer3_i = self._make_layer(BasicBlock, planes * 2, planes * 4, num_blocks=n, stride=2)
        self.layer4_i = self._make_layer(BasicBlock, planes * 4, planes * 8, num_blocks=n, stride=2)
        self.layer5_i = self._make_layer(Bottleneck, planes * 8, planes * 8, num_blocks=2, stride=2)

        # P-branch

        self.compression3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
        ], name='pidnet_compression3')

        self.compression4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
        ], name='pidnet_compression4')

        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, num_blocks=m, stride=1)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, num_blocks=m, stride=1)
        self.layer5_p = self._make_layer(Bottleneck, planes * 2, planes * 2, num_blocks=1, stride=1)

        # D-branch

        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes, stride=1)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, num_blocks=1, stride=1)

            self.diff3 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ], name='pidnet_diff3')

            self.diff4 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ], name='pidnet_diff4')

            self.spp = PAPPM(ppm_planes, planes * 4)
            self.dfm = LightBag(planes * 4)

        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2, stride=1)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2, stride=1)
            
            self.diff3 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ], name='pidnet_diff3')

            self.diff4 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ], name='pidnet_diff4')

            self.spp = DAPPM(ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, num_blocks=1, stride=1)

        # Prediction Head

        if self.augment:
            self.seghead_p = SegmentHead(head_planes, num_classes, use_sigmoid=use_sigmoid)
            self.seghead_d = SegmentHead(planes, 1, use_sigmoid=use_sigmoid)

        self.final_layer = SegmentHead(head_planes, num_classes, use_sigmoid=use_sigmoid)


    def call(self, inputs):
        original_width = tf.shape(inputs)[2]
        original_height = tf.shape(inputs)[1]

        width_output = tf.shape(inputs)[2] // 8
        height_output = tf.shape(inputs)[1] // 8

        # I-stream

        x = self.conv1(inputs)
        x = self.layer1_i(x)
        x = self.relu(self.layer2_i(self.relu(x)))

        x_p = self.layer3_p(x) # p-branch
        x_d = self.layer3_d(x) # d-branch

        x = self.relu(self.layer3_i(x))
        x_p = self.pag3([x_p, self.compression3(x)])
        x_d += tf.image.resize(self.diff3(x), size=(height_output, width_output))

        if self.augment:
            temp_p = x_p

        x = self.relu(self.layer4_i(x))
        x_p = self.layer4_p(self.relu(x_p))
        x_d = self.layer4_d(self.relu(x_d))

        x_p = self.pag4([x_p, self.compression4(x)])
        x_d += tf.image.resize(self.diff4(x), size=(height_output, width_output))

        if self.augment:
            temp_d = x_d

        x_p = self.layer5_p(self.relu(x_p))
        x_d = self.layer5_d(self.relu(x_d))

        x = tf.image.resize(self.spp(self.layer5_i(x)), size=(height_output, width_output))
        x_dfm = self.dfm([x_p, x, x_d])

        x_p = self.final_layer(x_dfm)

        x_p = tf.image.resize(x_p, size=(original_height, original_width), name='output')

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_p, x_extra_d]
        else:
            return x_p

    def _make_single_layer(self, block, inplanes, planes, stride=1):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * block.expansion, kernel_size=1, strides=stride, use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ])

        layer = block(planes, stride=stride, downsample=downsample, no_relu=True)

        return layer

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * block.expansion, kernel_size=1, strides=stride, use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ])

        no_relu = False
        layers = []
        layers.append(block(planes, stride=stride, downsample=downsample, no_relu=no_relu))
        for i in range(1, num_blocks):
            if i == num_blocks - 1:
                no_relu = True
            layers.append(block(planes, stride=1, no_relu=no_relu))

        return tf.keras.Sequential(layers)



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

        edge_att = tf.sigmoid(d)

        return self.conv(edge_att * p + (1 - edge_att) * i)

