# references https://github.com/WongKinYiu/ScaledYOLOv4/blob/yolov4-large/models/export.py
# and models.py

from tool import darknet2pytorch
import torch
from tool.torch_utils import do_detect
import cv2
import numpy as np

model = darknet2pytorch.Darknet('yolov4-obj.cfg', inference=True)
model.load_state_dict(torch.load('yolov4-obj.pth'))

img = cv2.imread('output_SIDE1_12FT_0_0_200_20210118110957_image.jpg')
img = cv2.resize(img, (480, 480))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

use_cuda = True
if use_cuda:
    model.cuda()

# model.model[-1].export = True

# code from do_detect
model.eval()

if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
elif type(img) == np.ndarray and len(img.shape) == 4:
    img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
else:
    print("unknow image type")

if use_cuda:
    img = img.cuda()
img = torch.autograd.Variable(img)

output = model(img)

# try:
print('\nStarting TorchScript export with torch %s...' % torch.__version__)
f = 'yolov4-obj-torchscript.pt'
ts = torch.jit.trace(model, img)
ts.save(f)
print('TorchScript export success, saved as %s' % f)
# except Exception as e:
#     print('TorchScript export failure: %s' % e)