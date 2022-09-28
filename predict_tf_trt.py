import cv2
import argparse
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


import numpy as np
from models.model_builder import ModelBuilder
import timeit

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,
                    help="Evaluation batch size", default=1)
parser.add_argument("--num_classes",     type=int,
                    help="Model num classes", default=2)
parser.add_argument("--image_size",     type=tuple,
                    help="Model image size (input resolution)", default=(640, 360))
parser.add_argument("--video_dir",    type=str,
                    help="Dataset directory", default='/home/park/0708_capture/videos')
parser.add_argument("--video_result_dir", type=str,
                    help="Test result save directory", default='/home/park/0708_capture/videos/results/')
parser.add_argument("--checkpoint_dir", type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--converted_model_path", type=str,
                    help="Saved model weights directory", default='./checkpoints/export_path_trt/1/')

args = parser.parse_args()


if __name__ == '__main__':
    print('load_model')
    seg_model = tf.saved_model.load(args.converted_model_path, tags=[tag_constants.SERVING])

    # output_1 -> build 버전
    # tf.image.resize_11

    print('infer')
    infer = seg_model.signatures['serving_default']

    dummy_img = tf.zeros((1, 640, 360, 3))
    outputs = infer(dummy_img)
    print(outputs)
    output = outputs['tf.image.resize_11']


    # Camera
    frame_width = 1280
    frame_height = 720
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    idx = 1
    AVG_FPS = 0
    while cv2.waitKey(1) < 0:
        ret, frame = capture.read()
        
        frame = frame[40: 40+640, 640-180:640+180]
        
        
        
        # frame = frame[0:640, 120:120+360]
        # print(frame.shape)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # img = tf.image.resize(img, size=args.image_size,
        #         method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        
        img /= 255
        
        img = tf.expand_dims(img, axis=0)

        
        start_t = timeit.default_timer()
        outputs = infer(img)
        
        output = outputs['tf.image.resize_11']
        terminate_t = timeit.default_timer()
        
        DL_FPS = int(1./(terminate_t - start_t ))

        semantic_output = tf.math.argmax(output, axis=-1)
        semantic_output = tf.expand_dims(semantic_output, axis=-1)
        


        semantic_output = semantic_output[0]

        
        semantic_output = tf.image.resize(semantic_output, (640, 360), tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy().astype(np.uint8)
        frame *= semantic_output


        AVG_FPS += DL_FPS


        cv2.putText(frame, 'DL MODEL FPS : {0}'.format(str(int(AVG_FPS/idx))),(30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 50, 0), 2, cv2.LINE_AA)
        cv2.imshow("VideoFrame", frame)

        idx += 1

    capture.release()
    cv2.destroyAllWindows()
