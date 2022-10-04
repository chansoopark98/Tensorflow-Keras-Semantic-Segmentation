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
                    help="Model num classes", default=2)
parser.add_argument("--image_size",     type=tuple,
                    help="Model image size (input resolution)", default=(384, 216))
parser.add_argument("--video_dir",    type=str,
                    help="Dataset directory", default='/home/park/0708_capture/videos')
parser.add_argument("--video_result_dir", type=str,
                    help="Test result save directory", default='/home/park/0708_capture/videos/results/')
parser.add_argument("--checkpoint_dir", type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_name", type=str,
                    help="Saved model weights directory", default='/1004/_1004_r384x216_b16_e100_lr0.005_adam_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    # model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model()
    from models.model_zoo.PIDNet import PIDNet

    model = PIDNet(input_shape=(*args.image_size, 3), m=2, n=3, num_classes=args.num_classes,
                planes=32, ppm_planes=96, head_planes=128, augment=False, training=False).build()
    

    # from models.model_zoo.pidnet.pidnet import PIDNet
        
    # model = PIDNet(input_shape=(*args.image_size, 3), m=2, n=3, num_classes=args.num_classes,
    #                    planes=32, ppm_planes=96, head_planes=128, augment=False)
    # model.build((None, *args.image_size, 3))


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
        print(frame.shape)
        frame = frame[40: 40+384, 200:200+216]
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

        # semantic_output = tf.math.argmax(output, axis=-1)
        semantic_output = tf.expand_dims(output, axis=-1)
        # semantic_output = tf.image.resize(semantic_output, (640, 360)).numpy().astype(np.uint8)
        
        # resize_shape = frame.shape
        # semantic_output = tf.expand_dims(semantic_output, axis=-1)
        # semantic_output = tf.image.resize(semantic_output, (resize_shape[0], resize_shape[1]),
        #                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # r = semantic_output[:, :, 0]
        # g = semantic_output[:, :, 0]
        # b = semantic_output[:, :, 0]

        # draw_r = frame[:, :, 0]
        # draw_g = frame[:, :, 1]
        # draw_b = frame[:, :, 2]
        
        # for j in range(1,args.num_classes):
        #     draw_r = tf.where(r==j, color_map[j-1][0], draw_r)
        #     draw_g = tf.where(g==j, color_map[j-1][1], draw_g)
        #     draw_b = tf.where(b==j, color_map[j-1][2], draw_b)

        # draw_r = np.expand_dims(draw_r, axis=-1)
        # draw_g = np.expand_dims(draw_g, axis=-1)
        # draw_b = np.expand_dims(draw_b, axis=-1)

        # convert_rgb = np.concatenate([draw_r, draw_g, draw_b], axis=-1).astype(np.uint8)
        
        # convert_rgb = cv2.cvtColor(convert_rgb, cv2.COLOR_RGB2BGR)
        # convert_rgb = tf.image.resize(convert_rgb, (frame_height, frame_width),
        #                                 method=tf.image.ResizeMethod.BILINEAR)
        # convert_rgb = convert_rgb.numpy().astype(np.uint8)

        
        # output = semantic_output[0].numpy().astype(np.uint8) * 50
        
        terminate_t = timeit.default_timer()
        
        FPS = int(1./(terminate_t - start_t ))

        semantic_output = semantic_output[0]

        
        semantic_output = tf.image.resize(semantic_output, (384, 216), tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy().astype(np.uint8)
        frame *= semantic_output

        cv2.putText(frame, 'FPS : {0}'.format(str(FPS)),(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (200, 50, 0), 3, cv2.LINE_AA)
        cv2.imshow("VideoFrame", frame)

    capture.release()
    cv2.destroyAllWindows()
