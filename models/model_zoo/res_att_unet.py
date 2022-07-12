import tensorflow as tf
from tensorflow.keras.models import * 
from tensorflow.keras.layers import *

def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def Attention_block(input1, input2, filters):
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input1)
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input2)
    x1 = BatchNormalization()(x1)
    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = multiply([input2, psi])
    return out

def Recurrent_block(input, channel, t=2):
    for i in range(t):
        if i == 0:
            x = Conv2D(channel, kernel_size=(3, 3), strides=1, padding='same')(input)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        out = Conv2D(channel, kernel_size=(3, 3), strides=1, padding='same')(add([x, x]))
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
    return out

def RRCNN_block(input, channel, t=2):
    x1 = Conv2D(channel, kernel_size=(1, 1), strides=1, padding='same')(input)
    x2 = Recurrent_block(x1, channel, t=t)
    x2 = Recurrent_block(x2, channel, t=t)
    out = add([x1, x2])
    return out



def R2AttUNet(image_size, num_classes):
    input_tensor = Input(shape=(image_size[0], image_size[1], 3))

    t = 2
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = RRCNN_block(input_tensor, filters[0], t=t)

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = RRCNN_block(e2, filters[1], t=t)

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = RRCNN_block(e3, filters[2], t=t)

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = RRCNN_block(e4, filters[3], t=t)

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = RRCNN_block(e5, filters[4], t=t)

    d5 = up_conv(e5, filters[3])
    x4 =  Attention_block(d5, e4, filters[3])
    d5 = Concatenate()([x4, d5])
    d5 = conv_block(d5, filters[3])

    d4 = up_conv(d5, filters[2])
    x3 =  Attention_block(d4, e3, filters[2])
    d4 = Concatenate()([x3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    x2 =  Attention_block(d3, e2, filters[1])
    d3 = Concatenate()([x2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    x1 =  Attention_block(d2, e1, filters[0])
    d2 = Concatenate()([x1, d2])
    d2 = conv_block(d2, filters[0])

    output = Conv2D(num_classes, (3, 3), padding='same', name='output')(d2)

    # if when use cross entropy


    model = Model(inputs=[input_tensor], outputs=[output])

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.he_normal()
        if hasattr(layer, 'depthwise_initializer'):
            layer.depthwise_initializer = tf.keras.initializers.he_normal()

    return model