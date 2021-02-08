import json
import cv2
import random
import math

data_path = '../BBOX-LABELS/'

with open(data_path+'jsonlabels.txt') as f:
    data = json.load(f)

all_data = []
for entry in data:
    strs = [data_path+entry['name']]
    if isinstance(entry['boxes'][0], int):
        box = entry['boxes']
        top_left = [box[0], box[1]]
        bottom_right = [box[0]+box[2]-1, box[1]+box[3]-1]
        strs.append('{x1},{y1},{x2},{y2},1'.format(x1=top_left[0], y1=top_left[1], x2=bottom_right[0], y2=bottom_right[1]))
        out_str = ' '.join(strs)
        out_str += '\n'
        all_data.append(out_str)
    else:
        for box in entry['boxes']:
            top_left = [box[0], box[1]]
            bottom_right = [box[0]+box[2]-1, box[1]+box[3]-1]
            strs.append('{x1},{y1},{x2},{y2},1'.format(x1=top_left[0], y1=top_left[1], x2=bottom_right[0], y2=bottom_right[1]))
        out_str = ' '.join(strs)
        out_str += '\n'
        all_data.append(out_str)

# shuffle and split into training/test 80/20
random.shuffle(all_data)
split_idx = math.ceil(len(all_data) * 0.2)
test_data = all_data[0:split_idx]
train_data = all_data[split_idx:]

# # small train and test sets for testing
# test_data = all_data[0:41]
# train_data = all_data[41:241]

with open('train.txt', 'w') as fout:
    for line in train_data:
        fout.write(line)

with open('data/val.txt', 'w') as fout:
    for line in test_data:
        fout.write(line)