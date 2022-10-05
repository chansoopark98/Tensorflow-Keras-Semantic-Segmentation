import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.initializers import VarianceScaling, he_normal

class ModelBuilder(object):
    def __init__(self, image_size: tuple = (640, 360), num_classes: int = 3,
                use_weight_decay: bool = False, weight_decay: float = 0.00001):
        """
            Args:
                image_size         (tuple) : Model input resolution ([H, W])
                num_classes        (int)   : Number of classes to classify 
                                            (must be equal to number of last filters in the model)
                use_weight_decay   (bool)  : Use weight decay.
                weight_decay       (float) : Weight decay value.
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_weight_decay = use_weight_decay
        self.weight_decay = weight_decay\


        # self.kernel_initializer = VarianceScaling(scale=2.0, mode="fan_out",
        #                                           distribution="truncated_normal")
        self.kernel_initializer = he_normal()


    def classifier(self, x: tf.Tensor, num_classes: int = 19, upper: int = 4,
                   prefix: str = None, activation: str = None) -> tf.Tensor:
        """
            Segmentation 출력을 logits 형태로 만듭니다. Conv2D + Upsampling (resize)
            Args: 
                x            (tf.Tensor)  : Input tensor (segmentation model output)
                num_classes  (int)        : Number of classes to classify
                upper        (int)        : Upsampling2D layer extension factor
                prefix       (str)        : Set the final output graph name
                activation   (activation) : Set the activation function of the Conv2D layer.
                                            logits do not apply activation. ( Default : None)
                                         
            Returns:
                x            (tf.Tensor)  : Output tensor (logits output)
        """
        x = layers.Conv2D(num_classes, kernel_size=1, strides=1, activation=activation,
                          kernel_initializer=self.kernel_initializer)(x)

        x = layers.UpSampling2D(size=(upper, upper),
                                interpolation='bilinear',
                                name=prefix)(x)
        return x


    def build_model(self, model_name: str, training: True) -> models.Model:
        """
            Build the model (you can build your custom model separately here)
            Args: 
                model_name   (str)    : Model name to build
                training     (bool)   : Set whether to train the model.
                                         Initialize weights and set attenuation when set to 'True'
            Returns:
                model        (models.Model) : Return tf.keras.models.Model
        """
        # Load and Build model
        if model_name == 'pidnet':
            from models.model_zoo.PIDNet import PIDNet
            if training:
                augment_mode = True
            else:
                augment_mode = False
                
            model = PIDNet(input_shape=(*self.image_size, 3), m=2, n=3, num_classes=self.num_classes,
                           planes=32, ppm_planes=96, head_planes=128, augment=augment_mode, training=training).build()
        
        # Initialize weights and set attenuation when set to training mode is activate.
        if training:
            # Set weight initializers
            for layer in model.layers:
                  # for Convolution layer
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel_initializer = self.kernel_initializer
                if hasattr(layer, 'depthwise_initializer'):
                    layer.depthwise_initializer = self.kernel_initializer
                
                  # for BatchNormalization
                if hasattr(layer, 'beta_initializer'):
                    layer.beta_initializer = "zeros"
                    layer.gamma_initializer = "ones"


            # Set weight decay
            if self.use_weight_decay:
                for layer in model.layers:
                    if isinstance(layer, layers.Conv2D):
                        layer.add_loss(lambda layer=layer: regularizers.L2(
                            self.weight_decay)(layer.kernel))
                    elif isinstance(layer, layers.SeparableConv2D):
                        layer.add_loss(lambda layer=layer: regularizers.L2(
                            self.weight_decay)(layer.depthwise_kernel))
                        layer.add_loss(lambda layer=layer: regularizers.L2(
                            self.weight_decay)(layer.pointwise_kernel))
                    elif isinstance(layer, layers.DepthwiseConv2D):
                        layer.add_loss(lambda layer=layer: regularizers.L2(
                            self.weight_decay)(layer.depthwise_kernel))

                    if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                        layer.add_loss(lambda layer=layer: regularizers.L2(
                            self.weight_decay)(layer.bias))
        return model

if __name__ == '__main__':
    ModelBuilder(image_size=(224, 224), num_classes=3).build_model()