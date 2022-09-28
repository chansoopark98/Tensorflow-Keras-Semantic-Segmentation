import tensorflow as tf
import numpy as np
import argparse
import timeit
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size',     type=int,
                    help='Evaluation batch size', default=1)
parser.add_argument('--num_classes',     type=int,
                    help='Model num classes', default=2)
parser.add_argument('--image_size',     type=tuple,
                    help='Model image size (input resolution)', default=(640, 360))
parser.add_argument('--checkpoint_dir', type=str,
                    help='Setting the model storage directory', default='./checkpoints/')
parser.add_argument('--weight_name', type=str,
                    help='Saved model weights directory', default='/0919/_0919_test_human_seg_640x360_pidnet_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    interpreter = tf.lite.Interpreter('checkpoints/quant_models/quant_test.tflite')
    interpreter.allocate_tensors()

     # Set OpenCV Camera
    frame_width = 1280
    frame_height = 720
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while cv2.waitKey(1) < 0:
        ret, frame = capture.read()
        
        start_t = timeit.default_timer()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = tf.image.resize(frame, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.imagenet_utils.preprocess_input(x=img, mode='torch')
        
        img = tf.expand_dims(img, axis=0)

        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']

        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)

        semantic_output = tf.math.argmax(output, axis=-1)
        
        
        output = semantic_output[0].numpy().astype(np.uint8) * 50

        terminate_t = timeit.default_timer()
        
        FPS = int(1./(terminate_t - start_t ))

        cv2.putText(output, str(FPS),(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (200, 50, 0), 3, cv2.LINE_AA)
        cv2.imshow('VideoFrame', output)

        print(FPS)    
    capture.release()
    cv2.destroyAllWindows()
