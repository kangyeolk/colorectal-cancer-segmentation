import torch
from torchvision import transforms, datasets
from torch.utils import data
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from utils import to_np, to_tensor

def get_loader(config, data_type):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    if config.task == 'cls':
        if data_type == 'train': 
            data_set = ClsLoader(data_path=config.train_data_path,
                                 transform=transform)
                                
            data_loader = data.DataLoader(dataset=data_set,
                                          batch_size=config.train_batch_size,
                                          shuffle=True,
                                          num_workers=2)
        
        if data_type == 'val': 
            data_set = ClsLoader(data_path=config.val_data_path,
                                 transform=transform)
                                
            data_loader = data.DataLoader(dataset=data_set,
                                          batch_size=config.val_batch_size,
                                          shuffle=True,
                                          num_workers=2)
    elif config.task == 'seg':
        if data_type == 'train': 
            data_set = SegLoader(data_path=config.train_data_path,
                                transform=transform)
                                
            data_loader = data.DataLoader(dataset=data_set,
                                        batch_size=config.train_batch_size,
                                        shuffle=True,
                                        num_workers=2)
        
        if data_type == 'val': 
            data_set = SegLoader(data_path=config.val_data_path,
                                transform=transform)
                                
            data_loader = data.DataLoader(dataset=data_set,
                                        batch_size=config.val_batch_size,
                                        shuffle=True,
                                        num_workers=2)
                                        
    
    elif config.mode == 'test':
        pass

    return data_loader

class ClsLoader(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.image_path = os.path.join(self.data_path, 'image')

        # List up
        self.image_list = os.listdir(self.image_path)
        assert 'annotation.txt' in os.listdir(self.data_path), 'No annotation information'
        self.anno_info = dict()
        self.read_anno()

    def read_anno(self):
        with open(os.path.join(self.data_path, 'annotation.txt'), 'r') as f:
            for info in f:
                name, label = info.split()
                self.anno_info[name] = int(label)

    def __getitem__(self, index):
        
        image_name = self.image_list[index]
        image = Image.open(os.path.join(self.image_path, image_name))
        label = self.anno_info[image_name]

        # Transform image, label in forms
        if self.transform is not None:
            image = self.transform(image)
        label = to_tensor(label).long()
        
        return image, label

    def __len__(self):
        return len(self.image_list)


class SegLoader(Dataset):

    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.image_path = os.path.join(self.data_path, 'image')
        self.label_path = os.path.join(self.data_path, 'label')

        # Syncronize orders of image and label
        self.target_list = []
        self.read_target()
        self.image_list = [f for f in os.listdir(self.image_path) if f in self.target_list]
        self.image_list = sorted(self.image_list)
        self.label_list = sorted(os.listdir(self.label_path))

    def read_target(self):
        with open(os.path.join(self.data_path, 'annotation.txt'), 'r') as f:
            for info in f:
                name, label = info.split()
                if label == '1':
                    self.target_list.append(name)
             
    def __getitem__(self, index):
        
        image_name = self.image_list[index]
        label_name = self.label_list[index]

        image = Image.open(os.path.join(self.image_path, image_name))
        label = Image.open(os.path.join(self.label_path, label_name))

        # Transform image, label in forms
        if self.transform is not None:
            image = self.transform(image)
        label = to_tensor(to_np(label)).long()
        
        return image, label

    def __len__(self):
        return len(self.image_list)






"""
def rgb_to_label(x):
    to_np = np.array(x)
    print(to_np.shape)
    print(to_np)
    H, W, C = to_np.shape
    label = np.zeros((H, W))
    #FIXME
    for i in range(H):
        for j in range(W):
            if x[:, i ,j] == [0, 0, 0]:
                label[i, j] = 0
            elif x[:, i, j] == [200, 0, 0]:
                label[i, j] = 1
    
    return label
"""

if __name__ == '__main__':
    img = Image.open('../../../data/cancer_data/ori_patch256/train/image/P2_0039_(1.00,1792,13312,256,256).jpg')
    _img = np.array(img)
    l = Image.open('../../../data/cancer_data/ori_patch256/train/label/P2_0039_(1.00,1792,13312,256,256)-labels.png')
    _l = np.array(l)

    print(_img.shape, _l.shape)
    print(_l)