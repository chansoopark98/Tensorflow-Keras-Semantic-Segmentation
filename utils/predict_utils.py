import numpy as np
import cv2

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
