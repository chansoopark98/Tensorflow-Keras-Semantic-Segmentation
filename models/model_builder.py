import tensorflow as tf
from .model import unet

def base_model(image_size):
    model_input, model_output = unet(input_shape=(image_size[0], image_size[1], 3))
    return tf.keras.Model(model_input, model_output)