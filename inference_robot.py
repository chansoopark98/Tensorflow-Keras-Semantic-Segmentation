#!/usr/bin/env python3
from models.model_builder import semantic_model, segmentation_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from utils.pyrealsense_camera import RealSenseCamera
import rospy
from std_msgs.msg import Float32

tf.keras.backend.clear_session()

rospy.init_node('topic_publisher')
pub = rospy.Publisher('counter', Float32)
rate = rospy.Rate(60)


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

def sharpning(img):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, sharpen_kernel)
    return img

def predict_full_image(img):
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

    return output, yx_coords

def predict_roi_image(img):
    return img

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


if __name__ == '__main__':  
    model = segmentation_model(image_size=IMAGE_SIZE)
    roi_model = semantic_model(image_size=(128, 128))

    weight_name = '_0323_L-bce_B-16_E-100_Optim-Adam_best_iou'
    roi_weight_name = '_0330_roi-CE-B16-E100-C16-SWISH-ADAM_best_iou'
    model.load_weights(weight_name + '.h5')
    roi_model.load_weights(roi_weight_name + '.h5')
    
    img = tf.zeros([480, 640, 3])
    img = tf.image.resize(img, size=IMAGE_SIZE,
                    method=tf.image.ResizeMethod.BILINEAR)   
    img = tf.cast(img, dtype=tf.float32)
    img = preprocess_input(img, mode='torch')
    img = tf.expand_dims(img, axis=0)
    pred = model.predict_on_batch(img)


    roi_img = tf.zeros([128, 128, 3])
    roi_img = tf.cast(roi_img, dtype=tf.float32)
    roi_img = preprocess_input(roi_img, mode='torch')
    roi_img = tf.expand_dims(roi_img, axis=0)
    roi_pred = roi_model.predict_on_batch(roi_img)


    # 944622074360 D435i
    # f1181780 park pc l515
    cam = RealSenseCamera(device_id='f0350818', width=IMAGE_SIZE[1], height=IMAGE_SIZE[0], fps=30) #0003b661b825 # f0350818 # f1181780 # f1231507 # 944622074360
    cam.connect() 

    previous_yx = [0, 0]
    while True:
        x,y,w,h = 0, 0, 0, 0
        image_bundle = cam.get_image_bundle()
        rgb  = image_bundle['rgb']
        
        # Inference
        img = tf.image.resize(rgb, size=IMAGE_SIZE,
                        method=tf.image.ResizeMethod.BILINEAR)
                                        
        img = tf.cast(img, dtype=tf.float32)
        img = preprocess_input(img, mode='torch')
        img = tf.expand_dims(img, axis=0)
        pred = model.predict_on_batch(img)
        pred = np.where(pred>=1.0, 1, 0)
        result = pred[0]
        
        pred = pred[0]

        result= result[:, :, 0].astype(np.uint8)  * 255
        

        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circle_contour = []
        if len(contours) != 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area >= 1000:
                    circle_contour.append(contour)

            if len(circle_contour) != 0:
                x,y,w,h = cv2.boundingRect(circle_contour[0])
            

            if x != 0:
                cropped_input_img = rgb.copy()[y:y+h, x:x+w]
                ROI = tf.image.resize_with_crop_or_pad(cropped_input_img, 128, 128)
                ROI = tf.cast(ROI, dtype=tf.float32)
                ROI = preprocess_input(ROI, mode='torch')
                ROI = tf.expand_dims(ROI, axis=0)
                ROI_PRED = roi_model.predict_on_batch(ROI)
                ROI_PRED = tf.math.argmax(ROI_PRED, axis=-1)
                ROI_PRED = tf.cast(ROI_PRED[0], tf.int32)
                
                zero_img = np.zeros((rgb.shape[0], rgb.shape[1]))

                ROI_PRED = tf.where(ROI_PRED==2, 127, 0)
                
                ROI_PRED = crop_center(ROI_PRED.numpy(), w, h)
        
                            
                if 512+64 >= x >= 128-64:
                    if 352+64 >= y >= 128-64:

                        if y+h >= 480:
                            new_h = 480
                        else:
                            new_h = y+h
                        
                        if x+w >= 640:
                            new_w = 640
                        else:
                            new_w = x+w

                        # zero_img[y:new_h, x:new_w] = ROI_PRED
                        zero_img[y:y+ROI_PRED.shape[0], x:x+ROI_PRED.shape[1]] = ROI_PRED

                        yx_coords = np.mean(np.column_stack(np.where(zero_img == 127)),axis=0)
            
                        if np.isnan(yx_coords[0]) != True:
                            previous_yx = yx_coords
                            cv2.circle(rgb, (int(yx_coords[1]), int(yx_coords[0])), int(3), (0, 0, 255), 3, cv2.LINE_AA)
        
        rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.imshow('Output', rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # print('x :', previous_yx[1], 'y :', previous_yx[0])
        pub.publish(previous_yx[0])
       
