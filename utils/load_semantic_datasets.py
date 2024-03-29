from utils.predict_utils import PrepareCityScapesLabel
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union
import os

AUTO = tf.data.experimental.AUTOTUNE

class DataLoadHandler(object):
    def __init__(self, data_dir: str, dataset_name: str):
        """
        This class performs pre-process work for each dataset and load tfds.
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            dataset_name (str)   : Tensorflow dataset name (e.g: 'citiscapes')
        
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.__select_dataset()


    def __select_dataset(self):
        try:
            if self.dataset_name == 'cityscapes':
                self.dataset_list = self.__load_cityscapes()
                self.train_key = 'image_left'
                self.label_key = 'segmentation_label'

                # configuration cityscapes label tools
                self.cityscapes_tools = PrepareCityScapesLabel()

            elif self.dataset_name == 'full_semantic':
                self.dataset_list = self.__load_custom_dataset()
                self.train_key = 'rgb'
                self.label_key = 'gt'

            elif self.dataset_name == 'human_segmentation':
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
                 dataset_name: str = 'cityscapes', norm_type: str ='div'):
        """
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            image_size   (tuple) : Model input image resolution 
            batch_size   (int)   : Batch size
            dataset_name (str)   : Tensorflow dataset name (e.g: 'cityscapes')
            norm_type    (str)   : Set input image normalization type (e.g: 'torch')
        """
        # Configuration
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.norm_type = norm_type
        self.pi = 3.14
        super().__init__(data_dir=self.data_dir, dataset_name=self.dataset_name)


    def load_test(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
            Load functions for data set validation and testing tasks
            Args:
                sample       (dict)  : Dataset loaded through tfds.load().
        """
        img = tf.cast(sample[self.train_key], tf.float32)
        labels = tf.cast(sample[self.label_key], dtype=tf.int32)
        
        original_img = img

        if self.dataset_name == 'cityscapes':
            labels = self.cityscapes_tools.encode_cityscape_label(label=labels, mode='test')
        elif self.dataset_name == 'human_segmentation':
            labels = tf.where(labels>=1, 1, 0)

        if self.norm_type == 'tf':
            # Normalize the input image to 'tf' style (-1 ~ 1)
            img = preprocess_input(img, mode='tf')
        elif self.norm_type == 'torch':
            # Normalize the input image to 'torch' style (0 ~ 1 with mean, std)
            img = preprocess_input(img, mode='torch')
        else:
            # Normalize the input image (0 ~ 1)
            img /= 255.

        # Convert int type
        labels = tf.cast(labels, dtype=tf.int32)
        
        return (img, labels, original_img)


    @tf.function    
    def prepare_data(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
            Load RGB images and segmentation labels from the dataset.
            Options: 
                (1)   For cityscapes, convert 35 classes to 19 foreground classes
                    and 1 background class (total: 20).
            Args:
                sample    (dict)  : Dataset loaded through tfds.load().

            Returns:
                (img, labels) (dict) : Returns the image and label extracted from sample as a key value.
        """
        img = tf.cast(sample[self.train_key], dtype=tf.float32)
        labels = tf.cast(sample[self.label_key], dtype=tf.float32)
        
        return (img, labels)


    @tf.function
    def preprocess(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
            Dataset mapping function to apply to the train dataset.
            Various methods can be applied here, such as image resizing, random cropping, etc.
            Args:
                sample    (dict)  : Dataset loaded through tfds.load().
            
            Returns:
                (img, labels) (dict) : tf.Tensor
        """
        img, labels = self.prepare_data(sample)
    
        if tf.random.uniform([]) > 0.5:
            img = tf.image.resize(img, size=(self.image_size[0], self.image_size[1]),
                                method=tf.image.ResizeMethod.BILINEAR)
            labels = tf.image.resize(labels, size=(self.image_size[0], self.image_size[1]),
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        else:
            if tf.random.uniform([]) > 0.2:
                img = tf.image.resize_with_crop_or_pad(img, self.image_size[0], self.image_size[1])
                labels = tf.image.resize_with_crop_or_pad(labels, self.image_size[0], self.image_size[1])

            else:
                scale = tf.random.uniform([], 1.05, 1.3)

                new_h = self.image_size[0] * scale
                new_w = self.image_size[1] * scale

                img = tf.image.resize(img, size=(new_h, new_w),
                                    method=tf.image.ResizeMethod.BILINEAR)
                labels = tf.image.resize(labels, size=(new_h, new_w),
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                
                concat_img = tf.concat([img, labels], axis=-1)
                concat_img = tf.image.random_crop(
                    concat_img, (self.image_size[0], self.image_size[1], 4))

                img = concat_img[:, :, :3]
                labels = concat_img[:, :, 3:]

        return (img, labels)
        

    @tf.function
    def augmentation(self, img: tf.Tensor, labels: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]: 
        """
            This is a data augmentation function to be applied to the train dataset.
            You can add a factor or augmentation method to be applied to each batch.     
            Args:
                img       (tf.Tensor)  : tf.Tensor data (shape=H,W,3)
                labels    (tf.Tensor)  : tf.Tensor data (shape=H,W,1)

            Returns:
                img       (tf.Tensor)  : tf.Tensor data (shape=H,W,3)
                labels    (tf.Tensor)  : tf.Tensor data (shape=H,W,1)
        """
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_jpeg_quality(img, 30, 100)

        if tf.random.uniform([]) > 0.2:
            # Degrees to Radian
            upper = 35 * (self.pi/180.0)

            rand_degree = tf.random.uniform([], minval=0., maxval=upper)

            img = tfa.image.rotate(img, rand_degree, interpolation='bilinear')
            labels = tfa.image.rotate(labels, rand_degree, interpolation='nearest')

        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_saturation(img, 0.9, 3)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_brightness(img, 60)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_contrast(img, 0.5, 2)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            labels = tf.image.flip_left_right(labels)
        if tf.random.uniform([]) > 0.1:
            channels = tf.unstack (img, axis=-1)
            img = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        
        # Label normalization (adjust to number of classes to classify)
        if self.dataset_name == 'cityscapes':
            labels = self.cityscapes_tools.encode_cityscape_label(label=labels, mode='test')
        elif self.dataset_name == 'human_segmentation':
            labels = tf.where(labels>=1, 1, 0)

        # Input image normalization
        if self.norm_type == 'tf':
            # Normalize the input image to 'tf' style (-1 ~ 1)
            img = preprocess_input(img, mode='tf')
        elif self.norm_type == 'torch':
            # Normalize the input image to 'torch' style (0 ~ 1 with mean, std)
            img = preprocess_input(img, mode='torch')
        else:
            # Normalize the input image (0 ~ 1)
            img /= 255.
        
        # Covnert to float32 label to int32 label
        # labels = tf.cast(labels, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.float32)

        return (img, labels)


    @tf.function
    def preprocess_valid(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
            This is a data processing mapping function to be used in the validation step during training.
            Args:
                sample    (dict)  : Dataset loaded through tfds.load().

            Returns:
                img       (tf.Tensor)  : tf.Tensor data (shape=H,W,3)
                labels    (tf.Tensor)  : tf.Tensor data (shape=H,W,1)
        """     
        img, labels = self.prepare_data(sample)

        img = tf.image.resize(img, size=(self.image_size[0], self.image_size[1]),
                              method=tf.image.ResizeMethod.BILINEAR)
        labels = tf.image.resize(labels, size=(self.image_size[0], self.image_size[1]),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Label normalization (adjust to number of classes to classify)
        if self.dataset_name == 'cityscapes':
            labels = self.cityscapes_tools.encode_cityscape_label(label=labels, mode='test')
        elif self.dataset_name == 'human_segmentation':
            labels = tf.where(labels>=1, 1, 0)

        # Input image normalization
        if self.norm_type == 'tf':
            # Normalize the input image to 'tf' style (-1 ~ 1)
            img = preprocess_input(img, mode='tf')
        elif self.norm_type == 'torch':
            # Normalize the input image to 'torch' style (0 ~ 1 with mean, std)
            img = preprocess_input(img, mode='torch')
        else:
            # Normalize the input image (0 ~ 1)
            img /= 255.

        # Covnert to float32 label to int32 label
        # labels = tf.cast(labels, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.float32)

        return (img, labels)


    def get_trainData(self, train_data: tf.data.Dataset) -> tf.data.Dataset:
        """
            Prepare the Tensorflow dataset (tf.data.Dataset)
            Args:
                train_data    (tf.data.Dataset)  : Dataset loaded through tfds.load().

            Returns:
                train_data    (tf.data.Dataset)  : Apply data augmentation, batch, and shuffling
        """    
        train_data = train_data.shuffle(256)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
        train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.prefetch(AUTO)
        train_data = train_data.repeat()
        return train_data


    def get_validData(self, valid_data: tf.data.Dataset) -> tf.data.Dataset:
        """
            Prepare the Tensorflow dataset (tf.data.Dataset)
            Args:
                valid_data    (tf.data.Dataset)  : Dataset loaded through tfds.load().

            Returns:
                valid_data    (tf.data.Dataset)  : Apply data resize, batch, and shuffling
        """    
        valid_data = valid_data.map(self.preprocess_valid, num_parallel_calls=AUTO)
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)
        return valid_data


    def get_testData(self, valid_data: tf.data.Dataset) -> tf.data.Dataset:
        """
            Prepare the Tensorflow dataset (tf.data.Dataset)
            Args:
                valid_data    (tf.data.Dataset)  : Dataset loaded through tfds.load().

            Returns:
                valid_data    (tf.data.Dataset)  : Apply data resize, batch, and shuffling
        """    
        valid_data = valid_data.map(self.load_test)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data