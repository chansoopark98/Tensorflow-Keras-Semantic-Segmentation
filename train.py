"""
    >>> python train.py run type 
    | Index |       Type        |     Required arguments |
    ------------------------------------------------------
    |   1.  | Vanilla training  |         None           |
    ------------------------------------------------------
    |   2.  | Convert .pb model | --saved_model_path     |
    |                             --saved_model          |
    ------------------------------------------------------
    
    >>> Run command
    # 1. sudo apt-get install libtcmalloc-minimal4
    # 2. check dir ! 
    # 3. dpkg -L libtcmalloc-minimal4
    # 4. LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py
"""

from model_configuration import ModelConfiguration
import tensorflow as tf
import argparse
import time

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()

# Set Convert to SavedMoel
parser.add_argument('--saved_model',  help='convert to SavedModel.pb', action='store_true')
parser.add_argument('--saved_model_path', type=str,   help='Saved model weight path',
                    default='./checkpoints/0927/_0927_new_pid_same_train_options_best_iou.h5')

# Set Training Options
parser.add_argument('--model_prefix',     type=str,    help='Model name',
                    default='Multi-adam-b8-e150-lr0.005-focal2.0-augment-boundary20_aux0.4-weightDecay')
parser.add_argument('--batch_size',       type=int,    help='Batch size per each GPU',
                    default=8)
parser.add_argument('--epoch',            type=int,    help='Training epochs',
                    default=150)
parser.add_argument('--lr',               type=float,  help='Initial learning rate',
                    default=0.005)
parser.add_argument('--weight_decay',     type=float,  help='Set Weight Decay',
                    default=0.0001)
parser.add_argument('--num_classes',      type=int,    help='Set number of classes to classification(BG+FG)',
                    default=2)
parser.add_argument('--image_size',       type=tuple,  help='Set network input size',
                    default=(640, 360))
parser.add_argument('--network_name',     type=str,    help='Select segmentation network\
                                                            |   network_name    : description | \
                                                            [ 1. pidnet       : A Real-time Semantic Segmentation Network\
                                                                                Inspired from PID Controller ]',
                    default='pidnet')
parser.add_argument('--image_norm_type',  type=str,    help='Set RGB image nornalize format (tf or torch or no)\
                                                             [ 1. tf    : Rescaling RGB image -1 ~ 1 from imageNet ]\
                                                             [ 2. torch : Rescaling RGB image 0 ~ 1 from imageNet ]\
                                                             [ 3. else  : Rescaling RGB image 0 ~ 1 only divide 255 ]',
                    default='div')
parser.add_argument('--loss_type',        type=str,    help='Set Train loss function\
                                                             [ 1. ce    : SparseCategoricalCrossEntropy ]\
                                                             [ 2. focal : SparseCategoricalFocalLoss ]',
                    default='focal')
parser.add_argument('--optimizer',        type=str,    help='Set optimizer',
                    default='adam')
parser.add_argument('--use_weightDecay',  type=bool,   help='Whether to use weightDecay',
                    default=False)
parser.add_argument('--mixed_precision',  type=bool,   help='Whether to use mixed_precision',
                    default=True)
parser.add_argument('--model_name',       type=str,    help='Set the model name to save',
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))

# Set directory path (Dataset,  Dataset_type, Chekcpoints, Tensorboard)
parser.add_argument('--dataset_dir',      type=str,    help='Set the dataset download directory',
                    default='./datasets/')
parser.add_argument('--dataset_name',      type=str,    help='Set the dataset type (cityscapes, custom etc..)',
                    default='human_segmentation')
parser.add_argument('--checkpoint_dir',   type=str,    help='Set the model storage directory',
                    default='./checkpoints/')
parser.add_argument('--tensorboard_dir',  type=str,    help='Set tensorboard storage path',
                    default='tensorboard/')

# Set Distribute training (When use Single gpu)
parser.add_argument('--gpu_num',          type=int,    help='Set GPU number to use(When without distribute training)',
                    default=1)

# Set Distribute training (When use Multi gpu)
parser.add_argument('--multi_gpu',  help='Set up distributed learning mode', action='store_true')

args = parser.parse_args()
                

if __name__ == '__main__':
    """If when use debug"""
    # tf.config.run_functions_eagerly(True)
    
    if args.saved_model:
        model = ModelConfiguration(args=args)
        model.saved_model()

    else:
        """
            Single GPU training
        """
        if args.multi_gpu == False:
            tf.config.set_soft_device_placement(True)

            gpu_number = '/device:GPU:' + str(args.gpu_num)
            with tf.device(gpu_number):
                model = ModelConfiguration(args=args)
                model.train()

        else:
            """
                Multi GPU training
            """
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = ModelConfiguration(args=args, mirrored_strategy=mirrored_strategy)
                model.train()