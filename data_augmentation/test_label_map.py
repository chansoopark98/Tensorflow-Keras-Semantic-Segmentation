import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--label_map_path",     type=str,   help="labelmap path", default='./data_augmentation/raw_data/labelmap.txt')

args = parser.parse_args()

label_map_path = args.label_map_path

label_list = []

"""
labelsmap.txt example

# label:color_rgb:parts:actions
background:0,0,0::
display:255,0,124::
indoor_display:63,202,197::
"""
with open(label_map_path, "r") as f:
    lines = f.readlines()

    for idx, line in enumerate(lines):

        if idx != 0:
            class_idx = idx-1


            
            split_line = line.split(':')
            
            class_name = split_line[0]
            string_rgb = split_line[1]
            
            r, g, b = string_rgb.split(',')
            r = int(r)
            g = int(g)
            b = int(b)

            output = {'class_name': class_name,
                    'class_idx': class_idx,
                    'rgb':(r, g, b)}

            label_list.append(output)

print(len(label_list))




            