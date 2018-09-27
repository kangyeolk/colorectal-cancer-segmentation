import torch
from torchvision import transforms, datasets
from torch.utils import data
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

def get_loader(config, data_type):

    transform = transforms.Compose([
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        transforms.ToTensor()
    ])
       
    if data_type == 'train': 
        data_set = TrainLoader(data_path=config.train_data_path,
                               transfrom=transform)
                               
        data_loader = data.DataLoader(dataset=data_set,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_worker=2,
                                      last_drop=False)
    
    if data_type == 'val': 
        data_set = TrainLoader(data_path=config.val_data_path,
                               transfrom=transform)
                               
        data_loader = data.DataLoader(dataset=data_set,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_worker=2,
                                      last_drop=False)
    
    elif config.mode == 'test':
        pass

    return data_loader

class TrainLoader(Dataset):

    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        image_path = os.path.join(self.data_path, 'image')
        mask_path = os.path.join(self.data_path, 'mask')

        # Syncronize orders of image and mask
        self.image_list = sorted(image_path)
        self.mask_list = sorted(mask_path)
    
    def __getitem__(self, index):

        image = self.image_list[index]
        mask = self.mask_list[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, mask

    def __len__(self):
        return len(self.image_list)
    
    


