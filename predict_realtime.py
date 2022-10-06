import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import numpy as np
from models.model_builder import ModelBuilder
import timeit

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,
                    help="Evaluation batch size", default=1)
parser.add_argument("--num_classes",     type=int,
                    help="Model num classes", default=1)
parser.add_argument("--image_size",     type=tuple,
                    help="Model image size (input resolution)", default=(640, 360))
parser.add_argument("--video_dir",    type=str,
                    help="Dataset directory", default='/home/park/0708_capture/videos')
parser.add_argument("--video_result_dir", type=str,
                    help="Test result save directory", default='/home/park/0708_capture/videos/results/')
parser.add_argument("--checkpoint_dir", type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_name", type=str,
                    help="Saved model weights directory", default='/1005/_1005_r640x360_b16_e_90_lr0.005_adam-binary_test-multi-gpu_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    from models.model_zoo.PIDNet import PIDNet

    model = PIDNet(input_shape=(*args.image_size, 3), m=2, n=3, num_classes=args.num_classes,
                planes=32, ppm_planes=96, head_planes=128, augment=False, training=False).build()
    
    model.load_weights(args.checkpoint_dir + args.weight_name, by_name=True)
    model.summary()


    # Camera
    frame_width = 1280
    frame_height = 720
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while cv2.waitKey(1) < 0:
        ret, frame = capture.read()
        
        h, w = args.image_size
        frame = frame[40: 40+h, 200:200+w]
        print(frame.shape)
        start_t = timeit.default_timer()
        
        # frame = frame[0:640, 120:120+360]
        # print(frame.shape)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = tf.image.resize(img, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        # img = preprocess_input(x=img, mode='torch')
        img /= 255
        
        img = tf.expand_dims(img, axis=0)

        output = model.predict_on_batch(img)

        semantic_output = tf.expand_dims(output, axis=-1)
        
        
        terminate_t = timeit.default_timer()
        
        FPS = int(1./(terminate_t - start_t ))

        semantic_output = semantic_output[0]

        
        semantic_output = tf.image.resize(semantic_output, (h, w), tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy().astype(np.uint8)
        frame *= semantic_output

        cv2.putText(frame, 'FPS : {0}'.format(str(FPS)),(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (200, 50, 0), 3, cv2.LINE_AA)
        cv2.imshow("VideoFrame", frame)

    capture.release()
    cv2.destroyAllWindows()
