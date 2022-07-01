import argparse
import time
import tensorflow as tf
from model_configuration import ModelConfiguration
# 1. sudo apt-get install libtcmalloc-minimal4
# 2. check dir ! 
# dpkg -L libtcmalloc-minimal4
# 3. LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()

# Set Convert to SavedMoel
parser.add_argument("--saved_model",  help="SavedModel.pb 변환", action='store_true')
parser.add_argument("--saved_model_path", type=str,   help="저장된 모델 가중치 경로",
                    default='./checkpoints/0629/_0629_224-224_16_100_0.001_adam_single_DDRNet_best_iou.h5')

# Set Training Options
parser.add_argument("--model_prefix",     type=str,    help="Model name",
                    default='Your_model_name')
parser.add_argument("--batch_size",       type=int,    help="배치 사이즈값 설정",
                    default=16)
parser.add_argument("--epoch",            type=int,    help="에폭 설정",
                    default=100)
parser.add_argument("--lr",               type=float,  help="Learning rate 설정",
                    default=0.001)
parser.add_argument("--weight_decay",     type=float,  help="Weight Decay 설정",
                    default=0.0005)
parser.add_argument("--num_classes",      type=int,    help="분류할 클래수 개수 설정",
                    default=2)
parser.add_argument("--image_size",       type=tuple,  help="조정할 이미지 크기 설정",
                    default=(224, 224))
parser.add_argument("--optimizer",        type=str,    help="Optimizer",
                    default='adamW')
parser.add_argument("--use_weightDecay",  type=bool,   help="weightDecay 사용 유무",
                    default=False)
parser.add_argument("--mixed_precision",  type=bool,   help="mixed_precision 사용",
                    default=False)
parser.add_argument("--model_name",       type=str,    help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))

# Set directory path (Dataset, Chekcpoints, Tensorboard)
parser.add_argument("--dataset_dir",      type=str,    help="데이터셋 다운로드 디렉토리 설정",
                    default='./datasets/')
parser.add_argument("--checkpoint_dir",   type=str,    help="모델 저장 디렉토리 설정",
                    default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,    help="텐서보드 저장 경로",
                    default='tensorboard/')

# Set Distribute training (Multi gpu)
parser.add_argument("--multi_gpu",  help="분산 학습 모드 설정", action='store_true')

args = parser.parse_args()
                

if __name__ == '__main__':
    if args.saved_model:
        model = ModelConfiguration(args=args)
        model.saved_model()

    else:
        if args.multi_gpu == False:
            tf.config.set_soft_device_placement(True)

            with tf.device('/device:GPU:0'):
                model = ModelConfiguration(args=args)
                model.train()

        else:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = ModelConfiguration(args=args, mirrored_strategy=mirrored_strategy)
                model.train()
