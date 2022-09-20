import tensorflow_datasets as tfds
import resource
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",    type=str,   help="Set the dataset download directory",
                    default='./datasets/')
parser.add_argument("--train_dataset",  type=str,   help="Set the dataset to be used for training (voc | coco)",
                    default='wider_face')

args = parser.parse_args()

if __name__ == '__main__':
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    os.makedirs(args.dataset_dir, exist_ok=True)


    train_data = tfds.load('coco/2017_panoptic', data_dir=args.dataset_dir, split='train')
    valid_data = tfds.load('coco/2017_panoptic', data_dir=args.dataset_dir, split='validation')


    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()      
    
    
    print("Nuber of train dataset = {0}".format(number_train))
    print("Nuber of validation dataset = {0}".format(number_valid))