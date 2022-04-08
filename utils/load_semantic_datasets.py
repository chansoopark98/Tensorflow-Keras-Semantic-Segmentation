from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

AUTO = tf.data.experimental.AUTOTUNE

def imgNetNorm(img):
    # img = tf.image.resize(img, size=(self.image_size[0], self.image_size[1]),
    #                 method=tf.image.ResizeMethod.BILINEAR)
    # mask = tf.image.resize(mask, size=(self.image_size[0], self.image_size[1]),
    #                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = img / 255.0
    img -= [0.485, 0.456, 0.406]  # imageNet mean
    img /= [0.229, 0.224, 0.225]  # imageNet std
    return img

@tf.function
def zoom(self, x, labels, scale_min=0.6, scale_max=1.6):
    h, w, _ = x.shape
    scale = tf.random.uniform([], scale_min, scale_max)
    nh = h * scale
    nw = w * scale
    x = tf.image.resize(x, (nh, nw), method=tf.image.ResizeMethod.BILINEAR)
    labels = tf.image.resize(labels, (nh, nw), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.image.resize_with_crop_or_pad(x, h, w)
    labels = tf.image.resize_with_crop_or_pad(labels, h, w)
    return x, labels

@tf.function
def rotate(self, x, labels, angle=(-45, 45)):
    angle = tf.random.uniform([], angle[0], angle[1], tf.float32)
    theta = np.pi * angle / 180

    # x = tfa.image.rotate(x, theta, interpolation="bilinear")
    # labels = tfa.image.rotate(labels, theta)
    return (x, labels)


class SemanticGenerator:
    def __init__(self, data_dir, image_size, batch_size, mode, data_type):
        """
        Args:
            data_dir: 데이터셋 상대 경로 ( default : './datasets/' )
            image_size: 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
            mode: 사용할 데이터셋 모드 [train, validation, all]
            data_type: roi (Custom2), full (Custom3)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size

        self.data_type= data_type
            

        if mode == 'train':
            self.train_data, self.number_train = self._load_train_datasets()
        elif mode == 'all':
            self.data, self.number_all = self._load_all_datasets()
        else:
            self.valid_data, self.number_valid = self._load_valid_datasets()


    def _load_valid_datasets(self):
        
        valid_data = tfds.load('FullSemantic',
                               data_dir=self.data_dir, split='train[90%:]')

        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        print("검증 데이터 개수:", number_valid)
        return valid_data, number_valid

    def _load_train_datasets(self):
        
        train_data = tfds.load('FullSemantic',
                               data_dir=self.data_dir, split='train[:90%]')


        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        print("학습 데이터 개수", number_train)
        return train_data, number_train

    def _load_all_datasets(self):
        data = tfds.load('FullSemantic' + self.data_type,
                               data_dir=self.data_dir, split='train')


        number_all = data.reduce(0, lambda x, _: x + 1).numpy()
        print("전체 데이터 개수", number_all)
        return data, number_all


    def load_test(self, sample):
        original = sample['rgb']
        img = tf.cast(original, tf.float32)
        labels = tf.cast(sample['gt'], tf.int64)
        
        if self.data_type == 'roi':
            img = tf.image.resize_with_crop_or_pad(img, 128, 128)
            labels = tf.image.resize_with_crop_or_pad(labels, 128, 128)

        img = preprocess_input(img, mode='torch')

        return (img, labels, original)


    @tf.function
    def preprocess(self, sample):
        img = tf.cast(sample['rgb'], tf.float32)
        labels = tf.cast(sample['gt'], tf.int64)

        if self.data_type == 'roi':
            img = tf.image.resize_with_crop_or_pad(img, 128, 128)
            labels = tf.image.resize_with_crop_or_pad(labels, 128, 128)

        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_saturation(img, 0.5, 1.5)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_brightness(img, 0.05)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_contrast(img, 0.5, 1.5)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            labels = tf.image.flip_left_right(labels)

        img = preprocess_input(img, mode='torch')

        return (img, labels)

    @tf.function
    def preprocess_valid(self, sample):
        original = sample['rgb']
        img = tf.cast(original, tf.float32)
        labels = tf.cast(sample['gt'], tf.int64)

        if self.data_type == 'roi':
            img = tf.image.resize_with_crop_or_pad(img, 128, 128)
            labels = tf.image.resize_with_crop_or_pad(labels, 128, 128)

        img = preprocess_input(img, mode='torch')

        return (img, labels)

    def get_trainData(self, train_data):
        train_data = train_data.shuffle(1024)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.prefetch(AUTO)
        train_data = train_data.repeat()
        return train_data

    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.preprocess_valid, num_parallel_calls=AUTO)
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)
        return valid_data

    def get_testData(self, valid_data):
        valid_data = valid_data.map(self.load_test)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data
