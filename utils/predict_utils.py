from collections import namedtuple
import tensorflow as tf

class PrepareCityScapesLabel(object):
    def __init__(self):
        """
        This class helps you convert Cityscapes into 19 universally used classes.
        Args:
            None
        Ref:
            https://stackoverflow.com/questions/56650201/how-to-convert-35-classes-of-cityscapes-dataset-to-19-classes

        """
        self.cityscapes_name = namedtuple('Label', [
            'name',
            'id',
            'trainId',
            'category',
            'categoryId',
            'hasInstances',
            'ignoreInEval',
            'color',
        ])

        self.cityscapes_label = [
            #                   name                       id    trainId   category            catId     hasInstances   ignoreInEval   color
            self.cityscapes_name(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            self.cityscapes_name(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            self.cityscapes_name(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            self.cityscapes_name(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            self.cityscapes_name(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            self.cityscapes_name(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
            self.cityscapes_name(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
            self.cityscapes_name(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
            self.cityscapes_name(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
            self.cityscapes_name(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
            self.cityscapes_name(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
            self.cityscapes_name(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
            self.cityscapes_name(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
            self.cityscapes_name(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
            self.cityscapes_name(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
            self.cityscapes_name(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
            self.cityscapes_name(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
            self.cityscapes_name(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
            self.cityscapes_name(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
            self.cityscapes_name(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
            self.cityscapes_name(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
            self.cityscapes_name(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
            self.cityscapes_name(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
            self.cityscapes_name(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
            self.cityscapes_name(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
            self.cityscapes_name(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
            self.cityscapes_name(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
            self.cityscapes_name(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
            self.cityscapes_name(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
            self.cityscapes_name(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
            self.cityscapes_name(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
            self.cityscapes_name(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
            self.cityscapes_name(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
            self.cityscapes_name(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
            self.cityscapes_name(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
        ]

        self.trainable_list = self.__convert_to_19(label_list=self.cityscapes_label)
        self.trainabel_color_map = self.encode_cityscape_color(label_list=self.cityscapes_label)


    def __convert_to_19(self, label_list: list) -> dict:
        """
            Function to remove ignore class(255) of Cityscapes list
            Args:
                label_list    (list)  : Cityscapes label list

            Returns:
                label_list    (dict)  : List of labels with background removed
        """    
        trainable_list = {}

        for label in label_list:
            if label.trainId == 255:
                trainable_list[label.id] = 0
                
            elif label.trainId == -1:
                trainable_list[label.id] = 0
            else:
                current_id = label.trainId + 1
                trainable_list[label.id] = current_id
        
        return trainable_list


    def encode_cityscape_label(self, label: tf.Tensor, mode: str = 'train') -> tf.Tensor:
        """
            Function to remove ignore class from Cityscape label
            Args:
                label  (tf.Tensor) : Labels (H, W, 1) returned by tfds.load() function
                mode   (str)       : 'train' or 'test', 'train' modes ignore background classes.

            Returns:
                label  (tf.Tensor) : Cityscape label with ignore class removed
        """   
        label_mask = tf.zeros_like(label, dtype=tf.int32)
        for k in self.trainable_list:
            label_mask = tf.where(label==k, self.trainable_list[k], label_mask)
        
        if mode == 'train':
            label_mask -= 1

        return label_mask

    def encode_cityscape_color(self, label_list: list) -> list:
        """
            Function to extract original colormap from label_list
            Args:
                label_list  (list) : Cityscapes label list
            
            Returns:
                encode_color_map  (list) : Encoded color map 
        """
        encode_color_map = []

        for label in label_list:
            if label.trainId != 255 and label.trainId != -1:
                encode_color_map.append(label.color)
        
        return encode_color_map
            

if __name__ == '__main__':
    cityscapes_prepare = PrepareCityScapesLabel()
    print(type(cityscapes_prepare.cityscapes_label))