import tensorflow as tf
import numpy as np
import cv2
import os
from model_configuration import ModelConfiguration

color_map = [
    (128, 64,128),
    (244, 35,232),
    # (127, 127, 127),
    # (255, 255, 255)
]

model_config = ModelConfiguration(args=None)
model = model_config.configuration_model(image_size=(224, 224), num_classes=3)

weightPath = os.path.join('/home/park/park/Tensorflow-Keras-Realtime-Segmentation/checkpoints/0707', "_0707_224_224-b16-e100-adam-ce_loss-new_data-ddrnet-single_best_iou.h5")
model.load_weights(weightPath)

path = '/home/park/0704_capture/videos'
filePath = os.path.join(path, "20220704_203042.mp4")
print(filePath)

if os.path.isfile(filePath):	# 해당 파일이 있는지 확인
    # 영상 객체(파일) 가져오기
    cap = cv2.VideoCapture(filePath)
else:
    print("파일이 존재하지 않습니다.")  

# 프레임을 정수형으로 형 변환
fps = cap.get(cv2.CAP_PROP_FPS)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임
 
frame_size = (frameWidth, frameHeight)
print('frame_size={}'.format(frame_size))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('/home/park/0704_capture/videos/test_output4.mp4',fourcc, fps, frame_size)
frame_idx = 0
while True:
    retval, frame = cap.read()
    
    frame_idx+=1
    if not(retval):
        break
    print(frame_idx)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = tf.image.resize(frame, size=(224, 224),
            method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.
    img = tf.expand_dims(img, axis=0)

    output = model.predict(img)
    output = tf.argmax(output, axis=-1)
    output = output[0]

    resize_shape = frame.shape
    output = tf.expand_dims(output, axis=-1)
    output = tf.image.resize(output, (resize_shape[0], resize_shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    

    r = output[:, :, 0]
    g = output[:, :, 0]
    b = output[:, :, 0]

    # cv2.imshow('test', output[:, :, 0].numpy().astype(np.uint8) * 127)
    # cv2.waitKey(int(fps))

    draw_r = frame[:, :, 0]
    draw_g = frame[:, :, 1]
    draw_b = frame[:, :, 2]
    

    for j in range(1,3):
        
        draw_r = tf.where(r==j, color_map[j-1][0], draw_r)
        draw_g = tf.where(g==j, color_map[j-1][1], draw_g)
        draw_b = tf.where(b==j, color_map[j-1][2], draw_b)

    draw_r = np.expand_dims(draw_r, axis=-1)
    draw_g = np.expand_dims(draw_g, axis=-1)
    draw_b = np.expand_dims(draw_b, axis=-1)

    convert_rgb = np.concatenate([draw_r, draw_g, draw_b], axis=-1).astype(np.uint8)
    
    convert_rgb = cv2.cvtColor(convert_rgb, cv2.COLOR_RGB2BGR)
    
    # cv2.imshow('test', convert_rgb)
    # cv2.waitKey(int(fps))

    out_video.write(convert_rgb)
        
    
        
if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
    cap.release()	# 영상 파일(카메라) 사용을 종료
    out_video.release()

