import os

def str2bool(x):
    return x.lower() in ['true', 'yes', 'y', 1]

def mkdir(x):
    if not os.path.exists(x):
        os.mkdir(x)
    