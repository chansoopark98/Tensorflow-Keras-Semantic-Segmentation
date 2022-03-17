import os
import glob
import cv2
from imageio import imread

font = cv2.FONT_HERSHEY_SIMPLEX


color_map =[
    (0, 0, 0),
    (111, 74, 0),
    (81, 0, 81),
    (128, 64, 128),
    (244, 35, 232),
    (230, 150, 140),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (150, 120, 90),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (52, 151, 52),
    (70, 130, 180),
    (220, 20, 60),
    (0, 0, 142),
    (0, 0, 230),
    (119, 11, 32)
    ]


def load_bbox(fname):
    grs = []
    points = []
    with open(fname) as f:
        while True:
            # Load 4 lines at a time, corners of bounding box.
            p0 = f.readline()
            if not p0:
                break  # EOF
            p1, p2, p3 = f.readline(), f.readline(), f.readline()
        
            points = [p0, p1, p2 ,p3 ]
            grs.append(points)
    
    return grs

def draw_grasps(img, grs):
    draw_img = img.copy()
    for grs_idx in range(len(grs)):
        grasps = grs[grs_idx]

        point1 = grasps[0].split()
        point2 = grasps[2].split()
        
        cv2.rectangle(draw_img, (int(float(point1[0])), int(float(point1[1]))), (int(float(point2[0])), int(float(point2[1]))), color_map[grs_idx], 1)
        cv2.putText(draw_img, str(grs_idx), (int(float(point1[0])), int(float(point1[1]))), font, 0.8,
                color_map[grs_idx], 3, cv2.LINE_AA)
    
    return draw_img

base_path = './for_annotation/'
file_path = 'demo_0227/'
save_path = base_path + file_path + '/corrected_annotation/'

os.makedirs(save_path, exist_ok=True)

# rgb_files = glob.glob(os.path.join(base_path + file_path + '/rgb/', '*rgb.png'))
# rgb_files = glob.glob(os.path.join(base_path + file_path + '/rgb/', '*.png'))
rgb_files = glob.glob(os.path.join(base_path + file_path + '*.png'))
rgb_files.sort()
# grasp_files = glob.glob(os.path.join(base_path + file_path + '/annotation/', '*_annotations.txt'))
# grasp_files = glob.glob(os.path.join(base_path + file_path + '/Annotations/', '*.txt'))
grasp_files = glob.glob(os.path.join(base_path + file_path + '*_annotations.txt'))
grasp_files.sort()

for idx in range(len(rgb_files)):
    img = imread(rgb_files[idx])
    
    grs = load_bbox(grasp_files[idx])
    fname = grasp_files[idx].split('/')
    print(fname)
    
    while True:
        
        draw_img = draw_grasps(img, grs)
        cv2.imshow('text', draw_img)
        key = cv2.waitKey(0)
        print(key)
        cv2.destroyAllWindows()
        
        delete_idx = abs(48 - key)
        
        if delete_idx == 65:
            break
        try:
            grs.pop(delete_idx)
        except:
            print('Out of range!!')

    with open(save_path + fname[3], 'w') as f:
        for i in range(len(grs)):
            for j in range(len(grs[i])):
                var = str(grs[i][j])
                
                f.writelines(var)
        f.close
        
    """
    0 - 48
    1 - 49
    q = 113

    delete_idx(q) = 65
    """
    

