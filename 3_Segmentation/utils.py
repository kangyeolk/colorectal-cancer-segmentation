import os
import numpy as np
import torch

def str2bool(x):
    return x.lower() in ['true', 'yes', 'y', 1]

def mkdir(x):
    if not os.path.exists(x):
        os.mkdir(x)

def move_files(curr, dest, file_list):
    for f in file_list:
        os.rename(os.path.join(curr, f),
                  os.path.join(dest, f))

def to_np(x):
    return np.array(x)

def to_tensor(x):
    return torch.tensor(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def cal_acc(preds, labels):
    """Calculate Classification Accuracy"""
    preds = torch.argmax(preds, dim=-1)
    out = (preds==labels).float()
    return torch.mean(out)

#TODO: convert multi-class to rgb value to saveeee... 
def label_to_rgb(x):
    pass

def cal_mIoU(preds, labels):
    """ 
    Calculate Intersection over Union
    args
        @preds: (N, C, H, W) dimensional output prediction maps
        @labels: (N, H, W) dimensional true class-labeled maps
    return
        @mIoU: vector, batch-wise
    """
    try:
        N, C, _, _ = preds.size()
    except:
        C, _, _ = preds.size()
        N = 1
    IoU = []
    labels_pred = torch.argmax(preds, dim=1)
    labels_pred = labels_pred.view(N, -1)
    labels = labels.view(N, -1)
    for i in range(C):
        inter_cnt = torch.sum(((labels_pred == labels) * (labels == i)).float(), dim=1)
        union_cnt = torch.sum((labels_pred == i).float()) \
                    + torch.sum((labels == i).float()) \
                    - inter_cnt
        IoU.append((inter_cnt / union_cnt))
    IoU.append((sum(IoU) / C)) # add mean IoU
    return IoU

def cal_pixel_acc(preds, labels):
    """ 
    Calculate Pixel Accuracy
    args 
        @preds: (N, C, H, W) dimensional output prediction maps
        @labels: (N, H, W) dimensional true class-labeled maps
    return
        @out: scalar, mean pixel accuaracy over all batch. 
    """
    labels_pred = torch.argmax(preds, dim=1)
    out = (labels_pred == labels).float() # ByteTensor => FloatTensor
    return torch.mean(out)
