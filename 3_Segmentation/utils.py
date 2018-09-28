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
