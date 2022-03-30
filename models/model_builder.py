from curses import use_default_colors
import tensorflow as tf
from .model import unet

def segmentation_model(image_size):
    model_input, model_output = unet(input_shape=(image_size[0], image_size[1], 3))
    return tf.keras.Model(model_input, model_output)

def semantic_model(image_size):
    model_input, model_output = unet(input_shape=(image_size[0], image_size[1], 3), base_channel=8, output_channel=3, use_logits=True)
    return tf.keras.Model(model_input, model_output)