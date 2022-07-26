from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Union
import os

AUTO = tf.data.experimental.AUTOTUNE


class DataLoadHandler:
    def __init__(self, data_dir: str, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.select_dataset()


    def select_dataset(self):
        try:
            if self.dataset_name == 'cityscapes':
                self.dataset_list = self.__load_cityscapes()
                self.train_key = 'image_left'
                self.label_key = 'segmentation_label'

            elif self.dataset_name == 'full_semantic':
                self.dataset_list = self.__load_custom_dataset()
                self.train_key = 'rgb'
                self.label_key = 'gt'
            else:
                raise Exception('Cannot find dataset_name! \n your dataset is {0}.'.format(
                    self.dataset_name))

            self.train_data, self.number_train, self.valid_data, self.number_valid = self.dataset_list
        
        except Exception as error:
            print('Cannot select dataset. \n {0}'.format(error))


    def __load_cityscapes(self):
        
        if os.path.isfile(self.data_dir + 'downloads/manual/leftImg8bit_trainvaltest.zip') == False:
            raise Exception('Download the \'leftImg8bit_trainvaltest.zip\' and \'gtFine_trainvaltest.zip\'' +
                            ' files from the Cityscape homepage and move the two compressed files to the' + 
                            ' $your_data_dir/downloads/manual/ directory. \n' +
                            'Check this page : https://www.tensorflow.org/datasets/catalog/cityscapes')

        train_data = tfds.load('cityscapes/semantic_segmentation',
                               data_dir=self.data_dir, split='train')
        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
            
        valid_data = tfds.load('cityscapes/semantic_segmentation',
                               data_dir=self.data_dir, split='validation')
        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()

        print("Nuber of train dataset = {0}".format(number_train))
        print("Nuber of validation dataset = {0}".format(number_valid))

        return (train_data, number_train, valid_data, number_valid)


    def __load_custom_dataset(self):
        train_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[10%:]')
        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        
        valid_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[:10%]')
        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()

        print("Nuber of train dataset = {0}".format(number_train))
        print("Nuber of validation dataset = {0}".format(number_valid))
        
        return (train_data, number_train, valid_data, number_valid)



    

class SemanticGenerator(DataLoadHandler):
    def __init__(self, data_dir: str, image_size: tuple, batch_size: int,
                 dataset_name: str = 'full_semantic'):
        """
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            image_size   (tuple) : Model input image resolution 
            batch_size   (int)   : Batch size
            dataset_name (str)   : Tensorflow dataset name (e.g: 'full_semantic')
        """
        # Configuration
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        super().__init__(data_dir=self.data_dir, dataset_name=self.dataset_name)
        print(self.train_key, self.label_key)


    def load_test(self, sample: dict):
        img = tf.cast(sample[self.train_key], dtype=tf.int32)
        labels = tf.cast(sample[self.label_key], dtype=tf.int32)

        if self.dataset_name == 'cityscapes':
            labels -= 1

        original_img = img

        # img = tf.image.resize(img, size=(self.image_size[0], self.image_size[1]),
        #                       method=tf.image.ResizeMethod.BILINEAR)
        # labels = tf.image.resize(labels, size=(self.image_size[0], self.image_size[1]),
        #                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
        img = preprocess_input(img, mode='tf')


        return (img, labels, original_img)


    @tf.function    
    def prepare_data(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        img = sample[self.train_key]
        labels = sample[self.label_key]

        if self.dataset_name == 'cityscapes':
            labels -= 1

        # convert to data type
        img = tf.cast(sample, dtype=tf.float32)
        labels = tf.cast(sample, dtype=tf.int32)

        return (img, labels)


    @tf.function
    def preprocess(self, sample: dict):
        img, labels = self.prepare_data(sample)

        img = tf.image.resize(img, size=(self.image_size[0], self.image_size[1]),
                              method=tf.image.ResizeMethod.BILINEAR)
        labels = tf.image.resize(labels, size=(self.image_size[0], self.image_size[1]),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if tf.random.uniform([]) > 0.5:
            scale = tf.random.uniform([], 0.8, 1.2)

            new_h = self.image_size[0] * scale
            new_w = self.image_size[1] * scale

            img = tf.image.resize(img, size=(new_h, new_w),
                                method=tf.image.ResizeMethod.BILINEAR)
            labels = tf.image.resize(labels, size=(new_h, new_w),
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            if scale >= 1.0:
                concat_img = tf.concat([img, labels], axis=-1)
                concat_img = tf.image.random_crop(
                    concat_img, (self.image_size[0], self.image_size[1], 4))

                img = concat_img[:, :, :3]
                labels = concat_img[:, :, 3:]

            else:
                img = tf.image.resize_with_crop_or_pad(img, self.image_size[0], self.image_size[1])
                labels = tf.image.resize_with_crop_or_pad(labels, self.image_size[0], self.image_size[1])

        return (img, labels)
        

    @tf.function
    def augmentation(self, img, labels):           
        if tf.random.uniform([]) > 0.8:
            img = tf.image.random_jpeg_quality(img, 30, 90)
        if tf.random.uniform([]) > 0.8:
            img = tf.image.random_saturation(img, 0.5, 1.5)
        if tf.random.uniform([]) > 0.8:
            img = tf.image.random_brightness(img, 32. / 255.)
        if tf.random.uniform([]) > 0.8:
            img = tf.image.random_contrast(img, 0.5, 1)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            labels = tf.image.flip_left_right(labels)

        img = preprocess_input(img, mode='tf')
        labels = tf.where(labels >= 1., 1., labels)
        confidence = tf.cast(tf.where(labels >= 1., 1., 0), dtype=tf.float32)

        gt = tf.concat([labels, confidence], axis=-1)

        return (img, gt)


    @tf.function
    def preprocess_valid(self, sample):
        img, labels = self.prepare_data(sample)

        img = tf.image.resize(img, size=(self.image_size[0], self.image_size[1]),
                              method=tf.image.ResizeMethod.BILINEAR)
        labels = tf.image.resize(labels, size=(self.image_size[0], self.image_size[1]),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img = preprocess_input(img, mode='tf')
        labels = tf.where(labels >= 1., 1., labels)
        confidence = tf.cast(tf.where(labels >= 1., 1., 0.), dtype=tf.float32)

        gt = tf.concat([labels, confidence], axis=-1)

        return (img, gt)


    def get_trainData(self, train_data):
        train_data = train_data.shuffle(512)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
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
