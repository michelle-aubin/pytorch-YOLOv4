from tool import darknet2pytorch
import torch
from tool.torch_utils import *

# load weights from darknet format
model = darknet2pytorch.Darknet('yolov4-obj.cfg', inference=True)
model.load_weights('yolov4-obj_best.weights')

# save weights to pytorch format
torch.save(model.state_dict(), 'yolov4-obj.pth')

# reload weights from pytorch format
model_pt = darknet2pytorch.Darknet('yolov4-obj.cfg', inference=True)
model_pt.load_state_dict(torch.load('yolov4-obj.pth'))