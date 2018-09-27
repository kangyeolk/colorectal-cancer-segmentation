import torch
import torch.nn as nn

import time

from model import UNet


class Solver:

    def __init__(config=config, train_loader=None, val_loader=None, test_loader=None):
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.cfg.mode == 'train':
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            self.test_loader = test_loader
        
        # Build model
        self.build_model()
        if self.cfg.pre_model:
            self.load_pre_model()

    
    def train(self):
        self.total_time = AverageMeter()
        self.loss = AverageMeter()
        self.IoU = AverageMeter()

        iter_per_epoch = len(self.train_loader) // self.cfg.batch_size
        if len(self.train_loader) % self.cfg.batch_size != 0:
            iter_per_epoch += 1
        
        for epoch in range(self.cfg.n_epochs):
            start_time = time.time()
            for i, (image, mask) in enumerate(self.train_loader):
                image_var = image.to(self.device)
                # NOTE: mask => pixel-wise 


    
    
    def build_model(self):
        """ Rough """
        self.model = UNet(num_classes=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.cfg.lr
                                      betas=(self.cfg.beta0, self.cfg.beta1))
    
    def load_pre_model(self):
        """ Load pretrained model """
        pass

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
