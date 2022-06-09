import tensorflow as tf
import tensorflow.keras.models as models
from .model_zoo.UNet import unet
from .model_zoo.DeepLabV3plus import DeeplabV3_plus
from .model_zoo.modify_DeepLabV3plus import deepLabV3Plus
from .model_zoo.EfficientNetV2 import EfficientNetV2S
from .model_zoo.DDRNet_23_slim import ddrnet_23_slim

def classifier(x, num_classes=19, upper=4, name=None):
    x = tf.keras.layers.Conv2D(num_classes, 1, strides=1,
                      kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_out", distribution="truncated_normal"))(x)
    x = tf.keras.layers.UpSampling2D(size=(upper, upper), interpolation='bilinear', name=name)(x)
    return x

def segmentation_model(image_size):
    # model_input, model_output = unet(input_shape=(image_size[0], image_size[1], 3), use_logits=False)
    # return tf.keras.Model(model_input, model_output)
    return ddrnet_23_slim(input_shape=(image_size[0], image_size[1], 3), num_classes=1)

    

def semantic_model(image_size):
    # model_input, model_output = unet(input_shape=(image_size[0], image_size[1], 3), base_channel=16, output_channel=3, use_logits=True)
    # return tf.keras.Model(model_input, model_output)

    base = EfficientNetV2S(input_shape=(image_size[0], image_size[1], 3), pretrained="imagenet")
    # base.load_weights('./checkpoints/efficientnetv2-s-imagenet.h5', by_name=True)")
    # base = EfficientNetV2M(input_shape=input_shape, pretrained="imagenet")

    base.summary()

    c5 = base.get_layer('add_34').output  # 16x32 256 or get_layer('post_swish') => 확장된 채널 1280
    c2 = base.get_layer('add_4').output  # 128x256 48
    """
    for EfficientNetV2S (input resolution: 512x1024)
    32x64 = 'add_34'
    64x128 = 'add_7'
    128x256 = 'add_4'
    """
    features = [c2, c5]

    model_input = base.input
    model_output = deepLabV3Plus(features=features, activation='swish')

    semantic_output = classifier(model_output, num_classes=2, upper=4, name='output')

    model = models.Model(inputs=[model_input], outputs=[semantic_output])
    
    return model