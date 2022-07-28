import numpy as np
import cv2
import tensorflow as tf
from collections import namedtuple


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
        label_mask = tf.zeros_like(label, dtype=tf.int32)
        for k in self.trainable_list:
            label_mask = tf.where(label==k, self.trainable_list[k], label_mask)
        
        if mode == 'train':
            label_mask -= 1

        return label_mask

    def encode_cityscape_color(self, label_list: list) -> list:
        encode_color_map = []

        for label in label_list:
            if label.trainId != 255 and label.trainId != -1:
                encode_color_map.append(label.color)
        
        return encode_color_map
            



color_map = [
    # (  0,  0,  0),
    (128, 64,128),
    (244, 35,232),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32)
]

def get_color_map(num_classes: int = 2):
    return color_map[:num_classes]


def project_p3d(p3d, cam_scale, K):
    p3d = p3d * cam_scale
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d


def draw_transform(img, trans, camera_intrinsic):
    arrow = np.array([[0,0,0],[0.05,0,0],[0,0.05,0],[0,0,0.05]])
    arrow = np.dot(arrow, trans[:3, :3].T) + trans[:3, 3]
    arrow_p2ds = project_p3d(arrow, 1.0, camera_intrinsic)
    img = draw_arrow(img, arrow_p2ds, thickness=2)
    return img



def draw_arrow(img, p2ds, thickness=1):
    h, w = img.shape[0], img.shape[1]
    for idx, pt_2d in enumerate(p2ds):
        p2ds[idx, 0] = np.clip(pt_2d[0], 0, w)
        p2ds[idx, 1] = np.clip(pt_2d[1], 0, h)
    
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[1,0], p2ds[1,1]), (0,0,255), thickness)
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[2,0], p2ds[2,1]), (0,255,0), thickness)
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[3,0], p2ds[3,1]), (255,0,0), thickness)
    return img


if __name__ == '__main__':
    cityscapes_prepare = PrepareCityScapesLabel()
    print(type(cityscapes_prepare.cityscapes_label))