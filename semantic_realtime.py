from models.model_builder import semantic_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from utils.pyrealsense_camera import RealSenseCamera

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
CHECKPOINT_DIR = args.checkpoint_dir
IMAGE_SIZE = (480, 640)
# IMAGE_SIZE = (720, 1280)
# IMAGE_SIZE = (128, 128)

def interlace(imgL, imgR, h, w):
    inter = np.empty((h, w, 3), imgL.dtype)
    inter[:h:2, :w, :] = imgL[:h:2, :w, :]
    inter[1:h:2, :w, :] = imgR[1:h:2, :w, :]

    return inter

if __name__ == '__main__':
    model = semantic_model(image_size=IMAGE_SIZE)
    weight_name = '_0329_CE-B16-E100-C16-RELU-ADAM_best_iou'
    # weight_name = '0330/_0330_roi-CE-B16-E100-C16-SWISH-ADAM_best_iou'
    model.load_weights(weight_name + '.h5')
    # null = tf.zeros([1,480, 640,3])
    # model.predict_on_batch(null)
    cam = RealSenseCamera(device_id='f0350818', width=IMAGE_SIZE[1], height=IMAGE_SIZE[0]) #0003b661b825 # f0350818 # f1181780 # f1231507
    cam.connect() 

    while True:
        image_bundle = cam.get_image_bundle()
        rgb  = image_bundle['rgb']
        # rrgb = image_bundle['rgb']
        
        # rgb = interlace(lrgb, rrgb,IMAGE_SIZE[0], IMAGE_SIZE[1])

        # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # rgb = cv2.filter2D(rgb, -1, sharpen_kernel)

        # depth = image_bundle['aligned_depth']
        img = tf.cast(rgb, dtype=tf.float32)
        img = preprocess_input(img, mode='torch')
        img = tf.expand_dims(img, axis=0)

        pred = model.predict_on_batch(img)
        pred = tf.argmax(pred[0], axis=-1)

        # Calculate x,y coordinates
        yx_coords = np.mean(np.column_stack(np.where(pred == 2)),axis=0)

        

        # For visualization
        output = pred.numpy() * 127
        output = output.astype(np.uint8)
        
        rgb =rgb.astype(np.uint8)
        rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        
        if np.isnan(yx_coords[0]) != True:
            cv2.circle(rgb, (int(yx_coords[1]), int(yx_coords[0])), int(3), (0, 0, 255), 3, cv2.LINE_AA)

        output = cv2.cvtColor(output,cv2.COLOR_GRAY2RGB)
        concat = cv2.hconcat([rgb, output])
        concat = cv2.resize(concat, dsize=(IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('test', concat)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
