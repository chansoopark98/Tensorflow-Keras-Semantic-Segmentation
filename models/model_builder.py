import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.models as models
from .model_zoo.UNet import unet
from .model_zoo.DeepLabV3plus import DeeplabV3_plus
from .model_zoo.modify_DeepLabV3plus import deepLabV3Plus
from .model_zoo.EfficientNetV2 import EfficientNetV2S
from .model_zoo.DDRNet_23_slim import ddrnet_23_slim
from .model_zoo.res_att_unet import R2AttUNet
from .model_zoo.mobileNetV3 import MobileNetV3_Small


def classifier(x, num_classes=19, upper=4, name=None):
    x = tf.keras.layers.Conv2D(num_classes,
                               kernel_size=1,
                               strides=1,
                               kernel_initializer=tf.keras.initializers.VarianceScaling(
                                   scale=1.0, mode="fan_out", distribution="truncated_normal")
                               )(x)
    x = tf.keras.layers.UpSampling2D(size=(upper, upper),
                                     interpolation='bilinear',
                                     name=name)(x)
    return x

def test_model(image_size):
    input_shape = (image_size[0], image_size[1], 3)
    inputs = Input(input_shape)
    

    x = SeparableConv2D(64, 3, 2, padding='same')(inputs)
    x = SeparableConv2D(64, 3, 1, padding='same')(x)
    x = SeparableConv2D(64, 1, 1, padding='same')(x)

    x = SeparableConv2D(128, 3, 2, padding='same')(x)
    x = SeparableConv2D(128, 3, 1, padding='same')(x)
    x = SeparableConv2D(128, 1, 1, padding='same')(x)

    x = SeparableConv2D(256, 3, 2, padding='same')(x)
    x = SeparableConv2D(256, 3, 1, padding='same')(x)
    x = SeparableConv2D(256, 1, 1, padding='same')(x)

    x = SeparableConv2D(512, 3, 2, padding='same')(x)
    x = SeparableConv2D(512, 3, 1, padding='same')(x)
    x = SeparableConv2D(512, 1, 1, padding='same')(x)
    

    # x = UpSampling2D(interpolation='bilinear')(x)
    # x = Conv2D(256, 1, 1, padding='same')(x)
    
    x = Conv2DTranspose(256, 3, 2, padding='same')(x)

    # x = UpSampling2D(interpolation='bilinear')(x)
    # x = Conv2D(128, 1, 1, padding='same')(x)
    x = Conv2DTranspose(128, 3, 2, padding='same')(x)

    # x = UpSampling2D(interpolation='bilinear')(x)
    # x = Conv2D(64, 1, 1, padding='same')(x)
    x = Conv2DTranspose(64, 3, 2, padding='same')(x)

    model = models.Model(inputs=[inputs], outputs=[x])

    return model
    
  

def semantic_model(image_size, model='MobileNetV3S', num_classes=2):
    """
    Args:
        image_size (tuple) : [B, H, W]
        model (str) : Model name
        num_classes (int) : number of classifer's
    """
    if model == 'MobileNetV3S':
        base = MobileNetV3_Small(shape=(
            image_size[0], image_size[1], 3), n_class=1000, alpha=1, include_top=False).build()
        c5 = base.get_layer('add_5').output
        c2 = base.get_layer('add').output  # 128x256 48
        features = [c2, c5]

        model_input = base.input
        model_output = deepLabV3Plus(features=features, activation='swish')

        semantic_output = classifier(
            model_output, num_classes=num_classes, upper=4, name='output')

        model = models.Model(inputs=[model_input], outputs=[semantic_output])

    elif model== 'EFFV2S':
        base = EfficientNetV2S(input_shape=(
            image_size[0], image_size[1], 3), pretrained='imagenet')
        c5 = base.get_layer('add_34').output
        c2 = base.get_layer('add_4').output

        features = [c2, c5]

        model_input = base.input
        model_output = deepLabV3Plus(features=features, activation='swish')

        semantic_output = classifier(
            model_output, num_classes=num_classes, upper=4, name='output')

        model = models.Model(inputs=[model_input], outputs=[semantic_output])


    elif model == 'ddrnet':
        model = ddrnet_23_slim(input_shape=[image_size[0], image_size[1], 3], num_classes=num_classes, use_aux=False)


    elif model == 'R2AttUNet':
        model = R2AttUNet(image_size=[image_size[0], image_size[1], 3], num_classes=num_classes)
    return model
