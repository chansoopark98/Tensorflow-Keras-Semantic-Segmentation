import tensorflow as tf
import tensorflow.keras.models as models
from .model_zoo.modify_DeepLabV3plus import deepLabV3Plus
from .model_zoo.EfficientNetV2 import EfficientNetV2S
from .model_zoo.DDRNet_23_slim import ddrnet_23_slim
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.initializers import VarianceScaling


class ModelBuilder():
    def __init__(self, image_size: tuple = (640, 480), num_classes: int = 3):
        """
        Args:
            image_size  (tuple) : Model input resolution ([H, W])
            num_classes (int)   : Number of classes to classify 
                                  (must be equal to number of last filters in the model)
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.kernel_initializer = VarianceScaling(scale=2.0, mode="fan_out",
                                                  distribution="truncated_normal")


    def classifier(self, x: tf.Tensor, num_classes: int = 19, upper: int = 4,
                   name: str = None, activation: str = None) -> tf.Tensor:

        x = Conv2D(num_classes, kernel_size=1, strides=1, activation=activation,
                   kernel_initializer=self.kernel_initializer)(x)

        x = UpSampling2D(size=(upper, upper),
                         interpolation='bilinear',
                         name=name)(x)
        return x


    def build_model(self) -> tf.keras.models.Model:
        """
        Build the model (you can build your custom model separately here)
        """
        base = EfficientNetV2S(input_shape=(
                self.image_size[0], self.image_size[1], 3), pretrained='imagenet')
        c5 = base.get_layer('add_34').output
        c2 = base.get_layer('add_4').output

        features = [c2, c5]

        model_input = base.input
        deeplab_output = deepLabV3Plus(features=features, activation='swish')

        semantic_output = self.classifier(
            deeplab_output, num_classes=self.num_classes, upper=4, name='semantic_output')

        confidence_output = self.classifier(
            deeplab_output, num_classes=1, upper=4, activation='sigmoid', name='confidence_output')

        model_output = Concatenate(name='output')([semantic_output, confidence_output])

        model = models.Model(inputs=[model_input], outputs=[model_output])
        
        # set weight initializers
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = self.kernel_initializer
            if hasattr(layer, 'depthwise_initializer'):
                layer.depthwise_initializer = self.kernel_initializer
        
        """
        When using a custom model, please use it in the form of a functional model
        (return is model input, model output) as shown below.
        
        # model_input, model_output = ddrnet_23_slim(input_shape=[
        #                                            self.image_size[0], self.image_size[1], 3],
        #                                            num_classes=self.num_classes, use_aux=False)
        # model = models.Model(inputs=model_input, outputs=model_output)
        """
        
        # model.su
        return model