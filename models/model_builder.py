import tensorflow as tf
from model import unet

def base_model(image_size, num_classes):
    model_input, model_output = unet(input_shape=(image_size[0], image_size[1], 1))
    return tf.keras.Model(model_input, model_output)