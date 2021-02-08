import json
import cv2
import random
import numpy as np
import math
import os




data_path = '../BBOX-LABELS/'
new_data_path = '../BBOX-LABELS-608/'
new_size = 608

os.mkdir(new_data_path)

with open(data_path+'jsonlabels.txt') as f:
    data = json.load(f)

all_data = []
for entry in data:
    strs = [new_data_path+entry['name']]
    img = cv2.imread(data_path+entry['name'])
    img = cv2.resize(img, (new_size, new_size))
    cv2.imwrite(new_data_path+entry['name'], img)
    scale = 608 / 480

    if isinstance(entry['boxes'][0], int):
        box = entry['boxes']
        top_left = [box[0], box[1]]
        bottom_right = [box[0]+box[2]-1, box[1]+box[3]-1]
        
        top_left[0] = int(np.round(float(top_left[0] * scale)))
        top_left[1] = int(np.round(float(top_left[1] * scale)))
        bottom_right[0] = int(np.round(float(bottom_right[0] * scale)))
        bottom_right[1] = int(np.round(float(bottom_right[1] * scale)))

        strs.append('{x1},{y1},{x2},{y2},0'.format(x1=top_left[0], y1=top_left[1], x2=bottom_right[0], y2=bottom_right[1]))
        out_str = ' '.join(strs)
        out_str += '\n'
        all_data.append(out_str)
    else:
        for box in entry['boxes']:
            top_left = [box[0], box[1]]
            bottom_right = [box[0]+box[2]-1, box[1]+box[3]-1]

            top_left[0] = int(np.round(float(top_left[0] * scale)))
            top_left[1] = int(np.round(float(top_left[1] * scale)))
            bottom_right[0] = int(np.round(float(bottom_right[0] * scale)))
            bottom_right[1] = int(np.round(float(bottom_right[1] * scale)))

            strs.append('{x1},{y1},{x2},{y2},0'.format(x1=top_left[0], y1=top_left[1], x2=bottom_right[0], y2=bottom_right[1]))
        out_str = ' '.join(strs)
        out_str += '\n'
        all_data.append(out_str)

# shuffle and split into training/test 80/20
random.shuffle(all_data)
split_idx = math.ceil(len(all_data) * 0.2)
test_data = all_data[0:split_idx]
train_data = all_data[split_idx:]

# small train and test sets for testing
# test_data = all_data[0:101]
# train_data = all_data[101:301]

with open(new_data_path+'train.txt', 'w') as fout:
    for line in train_data:
        fout.write(line)

with open(new_data_path+'val.txt', 'w') as fout:
    for line in test_data:
        fout.write(line)