import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils.predict_utils import get_color_map
import numpy as np
from models.model_builder import ModelBuilder

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
                    help="Test result save directory", default='/home/park/0708_capture/videos/results/')
parser.add_argument("--checkpoint_dir", type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_name", type=str,
                    help="Saved model weights directory", default='/0719/_0719_B8_E200_LR0.001_320-240_MultiGPU_sigmoid_activation_EFFV2S_best_iou.h5')

args = parser.parse_args()


if __name__ == '__main__':
    model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    color_map = get_color_map(num_classes=args.num_classes)

    # Camera
    frame_width = 480
    frame_height = 640
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = tf.image.resize(frame, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(x=img, mode='tf')
        
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
        
        for j in range(1,args.num_classes):
            draw_r = tf.where(r==j, color_map[j-1][0], draw_r)
            draw_g = tf.where(g==j, color_map[j-1][1], draw_g)
            draw_b = tf.where(b==j, color_map[j-1][2], draw_b)

        draw_r = np.expand_dims(draw_r, axis=-1)
        draw_g = np.expand_dims(draw_g, axis=-1)
        draw_b = np.expand_dims(draw_b, axis=-1)

        convert_rgb = np.concatenate([draw_r, draw_g, draw_b], axis=-1).astype(np.uint8)
        
        convert_rgb = cv2.cvtColor(convert_rgb, cv2.COLOR_RGB2BGR)
        convert_rgb = tf.image.resize(convert_rgb, (frame_height, frame_width),
                                        method=tf.image.ResizeMethod.BILINEAR)
        convert_rgb = convert_rgb.numpy().astype(np.uint8)


        cv2.imshow("VideoFrame", convert_rgb)

    capture.release()
    cv2.destroyAllWindows()
