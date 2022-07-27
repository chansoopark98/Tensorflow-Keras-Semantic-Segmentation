    
import tensorflow as tf
import os


def convert_to_trt(image_size: tuple, saved_model_path: str,
                   output_model_path: str, fp_mode: str = 'FP16'):

    os.makedirs(output_model_path, exist_ok=True)

    params = tf.experimental.tensorrt.ConversionParams(
                            precision_mode=fp_mode,
                            maximum_cached_engines=16)
    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=saved_model_path, conversion_params=params, use_dynamic_shape=False)
    converter.convert()

    def my_input_fn():
        # Define a generator function that yields input data, and use it to execute
        # the graph to build TRT engines.
        inp1 = tf.random.normal((1, image_size[0], image_size[1], 3), dtype=tf.float32)
        yield [inp1]

    # Generate corresponding TRT engines
    converter.build(input_fn=my_input_fn)
    # Generated engines will be saved.
    converter.save(output_model_path)