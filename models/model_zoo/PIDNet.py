import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from .pidnet.resnet import basic_block, bottleneck_block, basicblock_expansion, bottleneck_expansion
from .pidnet.model_utils import segmentation_head, DAPPPM, PAPPM, PagFM, Bag, Light_Bag

bn_mom = 0.1

class PIDNet(object):
    def __init__(self, input_shape=(640, 360, 3), m=2,
                n=3, num_classes=2, planes=64, ppm_planes=96, head_planes=128, augment=True, training=True):
        self.input_shape = input_shape
        self.m = m
        self.n = n
        self.num_classes = num_classes
        self.planes = planes
        self.ppm_planes = ppm_planes
        self.head_planes = head_planes
        self.augment = augment
        self.training = training

    
    def make_layer(self, x_in, block, inplanes, planes, blocks_num, stride=1, expansion=1, prefix='layer_name'):
        downsample = None
        if stride != 1 or inplanes != planes * expansion:
            downsample = layers.Conv2D((planes * expansion), kernel_size=(1, 1), strides=stride, use_bias=False, name=prefix + '_conv2d')(x_in)
            downsample = layers.BatchNormalization(momentum=bn_mom, name=prefix + '_bn')(downsample)
            # In original resnet paper relu was applied, But in pidenet it was not used
            # So commenting out for now
            # downsample = layers.Activation("relu")(downsample)

        x = block(x_in, planes, stride, downsample, prefix=prefix+'_basic_0')
        for i in range(1, blocks_num):
            
            if i == (blocks_num - 1):
                x = block(x, planes, stride=1, no_relu=True, prefix=prefix+'_basic_{0}'.format(i))
            else:
                x = block(x, planes, stride=1, no_relu=False, prefix=prefix+'_basic_{0}'.format(i))

        return x
        
    def build(self) -> models.Model:
        
        x_in = layers.Input(self.input_shape)

        # input_shape = tf.keras.backend.int_shape(x_in)
        height_output = self.input_shape[0] // 8
        width_output = self.input_shape[1] // 8

        # I Branch
        x = layers.Conv2D(self.planes, kernel_size=(3, 3), strides=2, padding='same')(x_in)
        x = layers.BatchNormalization(momentum=bn_mom)(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(self.planes, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization(momentum=bn_mom)(x)
        x = layers.Activation("relu")(x)

        x = self.make_layer(x, basic_block, self.planes, self.planes, self.m, expansion=basicblock_expansion, prefix='layer_1')  # layer1
        x = layers.Activation("relu")(x)

        x = self.make_layer(x, basic_block, self.planes, self.planes * 2, self.m, stride=2, expansion=basicblock_expansion, prefix='layer_2')  # layer2
        x = layers.Activation("relu")(x)

        x_ = self.make_layer(x, basic_block, self.planes * 2, self.planes * 2, self.m, expansion=basicblock_expansion, prefix='layer_3_0')  # layer3_
        if self.m == 2:
            x_d = self.make_layer(x, basic_block, self.planes * 2, self.planes, 0, expansion=basicblock_expansion, prefix='layer_3_d')  # layer3_d
        else:
            x_d = self.make_layer(x, basic_block, self.planes * 2, self.planes * 2, 0, expansion=basicblock_expansion, prefix='layer_3_d')  # layer3_d
        x_d = layers.Activation("relu")(x_d)

        x = self.make_layer(x, basic_block, self.planes * 2, self.planes * 4, self.n, stride=2, expansion=basicblock_expansion, prefix='layer_3')  # layer3
        x = layers.Activation("relu")(x)

        # P Branch
        compression3 = layers.Conv2D(self.planes * 2, kernel_size=(1, 1), use_bias=False)(x)  # compression3
        compression3 = layers.BatchNormalization(momentum=bn_mom)(compression3)

        x_ = PagFM(x_, compression3, self.planes * 2, self.planes)  # pag3

        if self.m == 2:
            diff3 = layers.Conv2D(self.planes, kernel_size=(3, 3), padding='same', use_bias=False)(x)  # diff3
            diff3 = layers.BatchNormalization(momentum=bn_mom)(diff3)
        else:
            diff3 = layers.Conv2D(self.planes * 2, kernel_size=(3, 3), padding='same', use_bias=False)(x)  # diff3
            diff3 = layers.BatchNormalization(momentum=bn_mom)(diff3)

        diff3 = tf.image.resize(diff3, size=(height_output, width_output), method='bilinear')
        
        # x_d = x_d + diff3
        x_d = tf.keras.layers.Add()([x_d, diff3])

        if self.augment:
            temp_p = x_

        layer4 = self.make_layer(x, basic_block, self.planes * 4, self.planes * 8, self.n, stride=2, expansion=basicblock_expansion, prefix='layer_4')  # layer4
        x = layers.Activation("relu")(layer4)

        x_ = layers.Activation("relu")(x_)
        x_ = self.make_layer(x_, basic_block, self.planes * 2, self.planes * 2, self.m, expansion=basicblock_expansion, prefix='layer_4_0')  # layer4_0

        x_d = layers.Activation("relu")(x_d)
        if self.m == 2:
            x_d = self.make_layer(x_d, bottleneck_block, self.planes, self.planes, 1, expansion=bottleneck_expansion, prefix='layer_4_d')  # layer4_d
        else:
            x_d = self.make_layer(x_d, basic_block, self.planes * 2, self.planes * 2, 0, expansion=basicblock_expansion, prefix='layer_4_d')  # layer4_d
            x_d = layers.Activation("relu")(x_d)

        compression4 = layers.Conv2D(self.planes * 2, kernel_size=(1, 1), use_bias=False)(x)  # compression4
        compression4 = layers.BatchNormalization(momentum=bn_mom)(compression4)
        x_ = PagFM(x_, compression4, self.planes * 2, self.planes)  # pag4

        diff4 = layers.Conv2D(self.planes * 2, kernel_size=(3, 3), padding='same', use_bias=False)(x)  # diff4
        diff4 = layers.BatchNormalization(momentum=bn_mom)(diff4)
        diff4 = tf.image.resize(diff4, size=(height_output, width_output), method='bilinear')
        # x_d = x_d + diff4
        x_d = tf.keras.layers.Add()([x_d, diff4])
        

        if self.augment:
            temp_d = x_d

        x_ = layers.Activation("relu")(x_)
        x_ = self.make_layer(x_, bottleneck_block, self.planes * 2, self.planes * 2, 1, expansion=bottleneck_expansion, prefix='layer_5_0')  # layer5_

        x_d = layers.Activation("relu")(x_d)
        x_d = self.make_layer(x_d, bottleneck_block, self.planes * 2, self.planes * 2, 1, expansion=bottleneck_expansion, prefix='layer_5_d')  # layer5_d

        layer5 = self.make_layer(x, bottleneck_block, self.planes * 8, self.planes * 8, 2, stride=2,
                            expansion=bottleneck_expansion, prefix='layer_5')  # layer5
        if self.m == 2:
            spp = PAPPM(layer5, self.ppm_planes, self.planes * 4)  # spp
            x = tf.image.resize(spp, size=(height_output, width_output), method='bilinear')
            dfm = Light_Bag(x_, x, x_d, self.planes * 4)  # dfm
        else:
            spp = DAPPPM(layer5,  self.ppm_planes, self.planes * 4)  # spp
            x = tf.image.resize(spp, size=(height_output, width_output), method='bilinear')
            dfm = Bag(x_, x, x_d, self.planes * 4)  # dfm

        x_ = segmentation_head(dfm, self.head_planes, self.num_classes, 8, prefix='main')  # final_layer

        # Prediction Head
        if self.augment:
            # aux loss
            seghead_p = segmentation_head(temp_p, self.head_planes, self.num_classes, 8, prefix='aux')

            # boundary loss
            seghead_d = segmentation_head(temp_d, self.planes, 1, 8, prefix='boundary')
            model_outputs = [seghead_p, x_, seghead_d]

            return models.Model(inputs=x_in, outputs=model_outputs)
        else:
            if self.training == False:
                x_ = tf.math.argmax(x_, axis=-1)
            return models.Model(inputs=x_in, outputs=x_)


if __name__ == '__main__':
    print('Test PIDNet model')

    model = PIDNet(input_shape=(480, 640, 3), m=2, n=3, num_classes=4,
                       planes=32, ppm_planes=96, head_planes=128, augment=False).build()
    
    model.summary()

