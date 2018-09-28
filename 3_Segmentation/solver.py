import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import time

from model import UNet

cudnn.benchmark = True

class Solver:

    def __init__(self, config, train_loader=None, val_loader=None, test_loader=None):
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

    
    def train_val(self):
        self.train_time = AverageMeter()
        self.train_loss = AverageMeter()
        self.IoU = AverageMeter() #TODO

        iter_per_epoch = len(self.train_loader) // self.cfg.batch_size
        if len(self.train_loader) % self.cfg.batch_size != 0:
            iter_per_epoch += 1
        
        for epoch in range(self.cfg.n_epochs):
            
            self.model.train()
            self.train_time.reset()
            self.train_loss.reset()
            self.IoU = AverageMeter()
            
            for i, (image, label) in enumerate(self.train_loader):
                start_time = time.time()
                image_var = image.to(self.device)
                label_var = label.to(self.device)
                
                output = self.model(image_var)
                # print(output.size(), label_var.size())
                loss = self.criterion(output, label_var)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                end_time = time.time()
                
                self.train_time.update(end_time - start_time)
                self.train_loss.update(loss)

                if (i + 1) % self.cfg.log_step == 0:
                    print('Epoch[{0}][{1}/{2}]\t'
                          'Time {train_time.val:.3f} ({train_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                              epoch + 1, i + 1, iter_per_epoch, 
                              train_time=self.train_time, loss=self.train_loss))

                if (epoch + 1) % self.cfg.val_step == 0:
                    self.validate(epoch)
        
    def validate(self, epoch):
        """ Validate with validation dataset """
        self.model.eval()
        pass
    
    
    def build_model(self):
        """ Rough """
        self.model = UNet(num_classes=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.cfg.lr,
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
