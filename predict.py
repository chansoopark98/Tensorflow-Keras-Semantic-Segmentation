from models.model_builder import segmentation_model
from utils.load_datasets import DatasetGenerator
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--result_dir", type=str,   help="Test result dir", default='./results/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
RESULT_DIR = args.result_dir
CHECKPOINT_DIR = args.checkpoint_dir
MASK_RESULT_DIR = RESULT_DIR + 'mask_result/'
IMAGE_SIZE = (480, 640)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MASK_RESULT_DIR, exist_ok=True)

TRAIN_INPUT_IMAGE_SIZE = IMAGE_SIZE
VALID_INPUT_IMAGE_SIZE = IMAGE_SIZE

test_dataset_config = DatasetGenerator(DATASET_DIR, TRAIN_INPUT_IMAGE_SIZE, BATCH_SIZE, mode='validation')
test_set = test_dataset_config.get_testData(test_dataset_config.valid_data)
test_steps = test_dataset_config.number_valid // BATCH_SIZE

model = segmentation_model(image_size=IMAGE_SIZE)

weight_name = '_0407_L-bce_B-16_E-100_Optim-Adam_Act-relu_best_iou'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')
model.summary()

batch_idx = 0
for x, y, original in tqdm(test_set, total=test_steps):
    pred = model.predict_on_batch(x)

    img = x[0]
    pred = pred[0]
    label = y[0]
    original = original[0]

    rows = 1
    cols = 4

    pred = tf.where(pred>=0.99, 1., 0)
    # plt.imshow(pred)
    # plt.show()
    # pred = tf.where(pred>=1.0, 1.0, 0.)
    
    original = tf.cast(original, tf.int32)
    output = (original * tf.cast(pred,tf.int32))

    fig = plt.figure()
    ax0 = fig.add_subplot(rows, cols, 1)
    ax0.imshow(original)
    ax0.set_title('img')
    ax0.axis("off")

    ax0 = fig.add_subplot(rows, cols, 2)
    ax0.imshow(label) 
    ax0.set_title('label')
    ax0.axis("off")

    ax1 = fig.add_subplot(rows, cols, 3)
    ax1.imshow(pred)
    ax1.set_title('pred')
    ax1.axis("off")

    ax1 = fig.add_subplot(rows, cols, 4)
    ax1.imshow(output)
    ax1.set_title('output')
    ax1.axis("off")

    batch_idx += 1
    tf.keras.preprocessing.image.save_img(MASK_RESULT_DIR + str(batch_idx) + 'output_mask.png', output)
    plt.savefig(RESULT_DIR + str(batch_idx) + '_output.png', dpi=300)