import cv2
import os
import glob
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir",    type=str,
                    help="Dataset directory", default='/home/park/0808_capture/video/trade_tower/')
parser.add_argument("--video_result_dir", type=str,
                    help="Test result save directory", default='/home/park/0808_capture/video/trade_tower/results/')

args = parser.parse_args()

if __name__ == '__main__':
    video_list = os.path.join(args.video_dir, '*.mp4')
    video_list = glob.glob(video_list)

    os.makedirs(args.video_result_dir, exist_ok=True)

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

        frame_idx = 0
        while True:
            print(frame_idx)
            retval, frame = cap.read()

            frame_idx+=1

            
            if not(retval):
                break
            if frame_idx % 50 == 0:
                cv2.imwrite(args.video_result_dir + '_' + str(video_idx) + '_' + str(frame_idx) + '.jpg', frame)
            


        if cap.isOpened():
            cap.release()
            

