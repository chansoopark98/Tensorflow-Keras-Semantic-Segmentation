import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import cv2
import os
from models.model_builder import ModelBuilder
import glob
from utils.predict_utils import get_color_map, draw_transform
from utils.draw_contours import find_and_draw_contours
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,
                    help="Evaluation batch size", default=1)
parser.add_argument("--num_classes",     type=int,
                    help="Model num classes", default=2)
parser.add_argument("--image_size",     type=tuple,
                    help="Model image size (input resolution)", default=(320, 240))
parser.add_argument("--video_dir",    type=str,
                    help="Dataset directory", default='/home/park/0708_capture/videos')
parser.add_argument("--video_result_dir", type=str,
                    help="Test result save directory", default='/home/park/0704_capture/videos/results/')
parser.add_argument("--checkpoint_dir", type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_name", type=str,
                    help="Saved model weights directory", default='/0722/_0722__0722_B8_E200_LR0.001_320-240_MultiGPU_sigmoid_activation_EFFV2S_scale_train100%_best_iou.h5')

args = parser.parse_args()

if __name__ == '__main__':
    video_list = os.path.join(args.video_dir, '*.mp4')
    video_list = glob.glob(video_list)

    os.makedirs(args.video_result_dir, exist_ok=True)

    model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    color_map = get_color_map(num_classes=args.num_classes)

    print(video_list)

    for video_idx, video_file in enumerate(video_list):
        video_idx += 1

        if os.path.isfile(video_file):	
            cap = cv2.VideoCapture(video_file)
        else:
            raise('cannot find file : {0}'.format(video_file))

        # Get camera FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30
        # Frame width size
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # Frame height size
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_size = (frameWidth, frameHeight)
        print('frame_size={0}'.format(frame_size))
        
        video_name = args.video_result_dir + str(video_idx) + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_name , fourcc, fps, frame_size)

        frame_idx = 0
        while True:
            print(frame_idx)
            retval, frame = cap.read()

            frame_idx+=1

            if not(retval):
                break
            
            original_frame_shape = frame.shape

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = tf.image.resize(frame, size=args.image_size,
                    method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, tf.float32)
            img = preprocess_input(x=img, mode='torch')
            
            img = tf.expand_dims(img, axis=0)

            output = model.predict(img)
            
            semantic_output = output[0, :, :, :args.num_classes]
            confidence_output = output[0, :, :, args.num_classes:]

            semantic_output = tf.argmax(semantic_output, axis=-1)

            resize_shape = frame.shape
            semantic_output = tf.expand_dims(semantic_output, axis=-1)
            semantic_output = tf.image.resize(semantic_output, (resize_shape[0], resize_shape[1]),
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
            r = semantic_output[:, :, 0]
            g = semantic_output[:, :, 0]
            b = semantic_output[:, :, 0]

            draw_r = frame[:, :, 0]
            draw_g = frame[:, :, 1]
            draw_b = frame[:, :, 2]
            
            # for j in range(1, args.num_classes):
            #     draw_r = tf.where(r==j, color_map[j-1][0], draw_r)
            #     draw_g = tf.where(g==j, color_map[j-1][1], draw_g)
            #     draw_b = tf.where(b==j, color_map[j-1][2], draw_b)

            draw_r = np.expand_dims(draw_r, axis=-1)
            draw_g = np.expand_dims(draw_g, axis=-1)
            draw_b = np.expand_dims(draw_b, axis=-1)

            convert_rgb = np.concatenate([draw_r, draw_g, draw_b], axis=-1).astype(np.uint8)
            
            convert_rgb = cv2.cvtColor(convert_rgb, cv2.COLOR_RGB2BGR)
            convert_rgb = tf.image.resize(convert_rgb, (original_frame_shape[0], original_frame_shape[1]),
                                          method=tf.image.ResizeMethod.BILINEAR)
            
            # convert to numpy array
            convert_rgb = convert_rgb.numpy().astype(np.uint8)
            semantic_output = semantic_output.numpy().astype(np.uint8) * 127
            
            # find and draw contours
            convert_rgb = find_and_draw_contours(img=convert_rgb, original_mask=semantic_output)


            # convert_rgb = convert_rgb.numpy().astype(np.uint8)
            video_writer.write(convert_rgb)
                
        video_writer.release()

        if cap.isOpened():
            cap.release()
            

