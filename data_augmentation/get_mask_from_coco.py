import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import cv2

rows = 1
cols = 2
dataset_dir = './datasets/'
train_data = tfds.load('coco/2017_panoptic', data_dir=dataset_dir, split='train')
valid_data = tfds.load('coco/2017_panoptic', data_dir=dataset_dir, split='validation')

data = train_data.concatenate(valid_data)

for sample in train_data.take(118287):

    img = sample['image']
    mask = sample['panoptic_image']

    fig = plt.figure()
    ax0 = fig.add_subplot(rows, cols, 1)
    ax0.imshow(img)
    ax0.set_title('original img')
    ax0.axis("off")

    ax0 = fig.add_subplot(rows, cols, 2)
    ax0.imshow(mask)
    ax0.set_title('mask')
    ax0.axis("off")

    plt.show()
    plt.close()



