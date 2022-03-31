from models.model_builder import semantic_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import argparse
import time
import cv2
import tensorflow as tf
from utils.realsense_camera import RealSenseCamera

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
# IMAGE_SIZE = (128, 128)

model = semantic_model(image_size=IMAGE_SIZE)

weight_name = '_0329_CE-B16-E100-C16-RELU-ADAM_best_iou'
# weight_name = '0330/_0330_roi-CE-B16-E100-C16-SWISH-ADAM_best_iou'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')


if __name__ == '__main__':
    cam = RealSenseCamera(device_id='f1181780') #0003b661b825 # f0350818 # f1181780
    cam.connect() 

    while True:
        image_bundle = cam.get_image_bundle()
        img  = image_bundle['rgb']
        depth = image_bundle['aligned_depth']

        img = tf.image.resize(img, size=IMAGE_SIZE,
                method=tf.image.ResizeMethod.BILINEAR)   
        img = tf.cast(img, dtype=tf.float32)
        img = preprocess_input(img, mode='torch')
        img = tf.expand_dims(img, axis=0)

        pred = model.predict_on_batch(img)
        pred = tf.argmax(pred, axis=-1)

        cv2.imshow('test', pred[0])
        




