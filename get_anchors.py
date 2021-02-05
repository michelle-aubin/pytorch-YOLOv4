import json
import cv2
import random
import math
from sklearn.cluster import KMeans
import numpy as np

def MyFn(lst):
    return lst[0] * lst[1]

data_path = '../BBOX-LABELS/'

with open(data_path+'jsonlabels.txt') as f:
    data = json.load(f)

all_w_h = []
for entry in data:
    if isinstance(entry['boxes'][0], int):
        box = entry['boxes']
        w_h = [box[2], box[3]]
        all_w_h.append(w_h)
    else:
        for box in entry['boxes']:
            w_h = [box[2], box[3]]
            all_w_h.append(w_h)


sample = random.sample(all_w_h, 1000)
sample = np.array(sample)
kmeans = KMeans(n_clusters=9).fit(sample)
rounded = []
for x in kmeans.cluster_centers_:
    rounded.append([round(x[0]), round(x[1])])
rounded = sorted(rounded, key=MyFn)
print(rounded)