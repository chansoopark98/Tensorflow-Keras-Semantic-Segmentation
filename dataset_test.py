import tensorflow as tf
import matplotlib.pyplot as plt
from utils.load_semantic_datasets import SemanticGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--dataset_type", type=str, help="테스트할 데이터셋 선택  'binary' or 'semantic'", default='semantic')
parser.add_argument("--dataset_nums", type=int, help="테스트 이미지 개수  'binary' or 'semantic'", default=100)
args = parser.parse_args()

DATASET_DIR = args.dataset_dir
DATASET_TYPE = args.dataset_type
DATASET_NUMS = args.dataset_nums
IMAGE_SIZE = (640, 480)

if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    train_dataset_config = SemanticGenerator(DATASET_DIR, IMAGE_SIZE, batch_size=1, dataset_name='cityscapes')
    train_data = train_dataset_config.get_testData(train_dataset_config.train_data)

    rows = 1
    cols = 3

    for img, mask, original in train_data.take(DATASET_NUMS):

        img = img[0]
        original = original[0]
        mask = mask[0]

        fig = plt.figure()
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img)
        ax0.set_title('img')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(mask)
        ax0.set_title('mask')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 3)
        ax0.imshow(original)
        ax0.set_title('original')
        ax0.axis("off")
        plt.show()
        plt.close()
