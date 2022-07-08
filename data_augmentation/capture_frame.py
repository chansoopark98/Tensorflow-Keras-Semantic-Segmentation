import numpy as np
import cv2
import os
import glob
import time

# 20220704_202231
# 20220704_202238
# 20220607_154917


video_path = '/home/park/0708_capture/videos'
video_list = os.path.join(video_path, '*.mp4')

save_video_path = video_path + '/result/'
os.makedirs(save_video_path, exist_ok=True)
    
video_list = glob.glob(video_list)

FPS = 3

video_idx = 0

for video_file in video_list:
    print(video_file)
    file_name = video_file.split('/')[5].split('.')[0]

    video_idx += 1
    if os.path.isfile(video_file):	# 해당 파일이 있는지 확인
        # 영상 객체(파일) 가져오기
        cap = cv2.VideoCapture(video_file)
    else:
        raise('cannot find file : {0}'.format(video_file))

    # 카메라 FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 영상의 넓이(가로) 프레임
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 영상의 높이(세로) 프레임
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    prev_time = 0
    frame_idx = 0
    while True:
        retval, frame = cap.read()
        
        frame_idx+=1

        if not(retval):
            break
        
        current_time = time.time() - prev_time

        if (retval is True) and (current_time > 1. / FPS):
            prev_time = time.time()

            # cv2.imshow('capture', frame)
            # cv2.waitKey(5000)

            cv2.imwrite(save_video_path + file_name + '_' +str(frame_idx) + '.jpg', frame)

        
            
    cv2.destroyAllWindows()
            
    if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
        cap.release()	# 영상 파일(카메라) 사용을 종료

