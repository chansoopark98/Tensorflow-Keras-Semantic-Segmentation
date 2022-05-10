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
parser.add_argument("--weight_name", type=str,   help="모델 저장 디렉토리 설정", default='weight_name.h5')
parser.add_argument("--serial_num", type=str,   help="모델 저장 디렉토리 설정", default='serial_num')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
CHECKPOINT_DIR = args.checkpoint_dir
WEIGHT_NAME = args.weight_name
REALSENSE_SIZE = (720, 1280)
CROP_SIZE = (480, 640)

if __name__ == '__main__':
    model = semantic_model(image_size=CROP_SIZE)
    model.load_weights(CHECKPOINT_DIR + WEIGHT_NAME)
    cam = RealSenseCamera(device_id='944622074360', width=REALSENSE_SIZE[1], height=REALSENSE_SIZE[0], fps=30) #0003b661b825 # f0350818 # f1181780 # f1231507
    cam.connect()

    image_bundle = cam.get_image_bundle()
    original  = image_bundle['rgb']
    rgb_shape = original.shape
    y_index = (rgb_shape[0] - CROP_SIZE[0]) // 2 
    x_index = (rgb_shape[1] - CROP_SIZE[1]) // 2

    while True:
        yx_coords = []
        previous_yx = []
        x_list = []
        y_list = []

        image_bundle = cam.get_image_bundle()
        original  = image_bundle['rgb']
        rgb = original.copy()[y_index:y_index+CROP_SIZE[0], x_index:x_index+CROP_SIZE[1]]

        # Inference                     
        img = tf.cast(rgb, dtype=tf.float32)
        img = preprocess_input(img, mode='torch')
        img = tf.expand_dims(img, axis=0)
        pred = model.predict_on_batch(img)
        pred = tf.math.argmax(pred, axis=-1)
        pred = tf.cast(pred, tf.uint8)
        
        result = pred[0]
        result = result.numpy()  * 127

        new_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        hole_result = np.where(result== 254,254, 0)
        hole_result = hole_result.astype(np.uint8)
        contours, _ = cv2.findContours(hole_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangle_contours = []
        if len(contours) != 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= 20:
                    rectangle_contours.append(contour)

            if len(rectangle_contours) != 0:
                area_mask = hole_result.copy()
                for contours_idx in range(len(rectangle_contours)):
                    x,y,w,h = cv2.boundingRect(rectangle_contours[contours_idx])
                    semantic_area = area_mask[y:y+h, x:x+w]
                                        
                    argwhere = np.argwhere(semantic_area == 254)
                    max_coord = np.max(argwhere, axis=0)
                    min_coord = np.min(argwhere, axis=0)

                    coord = (max_coord - min_coord) //2
            
                    yx_coords.append(coord)
                    x_list.append(x)
                    y_list.append(y)
                
                for i in range(len(yx_coords)):
                    if np.isnan(yx_coords[i][0]) != True: 
                        previous_yx.append([yx_coords[i][0] + y_list[i], yx_coords[i][1] + x_list[i]])
                
                original[y_index:y_index+CROP_SIZE[0], x_index:x_index+CROP_SIZE[1]] = rgb
                for i in range(len(previous_yx)):
                    cv2.circle(original, (int(previous_yx[i][1]+x_index), int(previous_yx[i][0])+y_index), int(3), (0, 0, 255), 3, cv2.LINE_AA)

        output = cv2.cvtColor(output,cv2.COLOR_GRAY2RGB)
        concat = cv2.hconcat([rgb, output])
        concat = cv2.resize(concat, dsize=(REALSENSE_SIZE[1], REALSENSE_SIZE[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('test', concat)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
