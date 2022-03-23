import tensorflow as tf
import matplotlib.pyplot as plt
from utils.load_datasets import DatasetGenerator
import argparse
from skimage import color

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
IMAGE_SIZE = (512, 512)

# train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)

if __name__ == "__main__":
    train_dataset_config = DatasetGenerator(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train')
    # train_dataset_config = Dataset(DATASET_DIR, IMAGE_SIZE, batch_size=1, mode='train', dataset='CustomCeleba')
    train_data = train_dataset_config.get_testData(train_dataset_config.train_data)

    for img, mask, original in train_data.take(10):


        rows = 1
        cols = 2
        fig = plt.figure()

        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img[0])
        ax0.set_title('rgb->lab->rgb')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(mask[0])
        ax0.set_title('original')
        ax0.axis("off")


        plt.show()
