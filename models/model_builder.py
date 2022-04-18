import tensorflow as tf
from .model_zoo.UNet import unet
from .model_zoo.DDRNet_23_slim import ddrnet_23_slim

def segmentation_model(image_size):
    # model_input, model_output = unet(input_shape=(image_size[0], image_size[1], 3), use_logits=False)
    # return tf.keras.Model(model_input, model_output)
    return ddrnet_23_slim(input_shape=(image_size[0], image_size[1], 3), num_classes=1)

    

def semantic_model(image_size):
    # model_input, model_output = unet(input_shape=(image_size[0], image_size[1], 3), base_channel=16, output_channel=3, use_logits=True)
    # return tf.keras.Model(model_input, model_output)
    return ddrnet_23_slim(input_shape=(image_size[0], image_size[1], 3), num_classes=3, use_aux=False)