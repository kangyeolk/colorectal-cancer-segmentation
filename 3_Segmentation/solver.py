import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

import time

from model import UNet
from utils import cal_mIoU, cal_pixel_acc, denorm
# from logger import Logger

cudnn.benchmark = True

class Solver:

    def __init__(self, config, train_loader=None, val_loader=None, test_loader=None):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpus = self.cfg.n_gpus

        if self.cfg.mode == 'train':
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            self.test_loader = test_loader
        
        # Build model
        self.build_model()
        if self.cfg.resume:
            self.load_pre_model()
        else:
            self.start_epoch = 0

    
    def train_val(self):
        # Build record objs
        self.build_recorder()
        
        iter_per_epoch = len(self.train_loader) // self.cfg.train_batch_size
        if len(self.train_loader) % self.cfg.train_batch_size != 0:
            iter_per_epoch += 1
        
        for epoch in range(self.start_epoch, self.start_epoch + self.cfg.n_epochs):
            
            self.model.train()
            
            self.train_time.reset()
            self.train_loss.reset()
            self.train_pix_acc.reset()
            self.train_mIoU.reset()
            
            for i, (image, label) in enumerate(self.train_loader):
                start_time = time.time()
                image_var = image.to(self.device)
                label_var = label.to(self.device)
                
                output = self.model(image_var)
                loss = self.criterion(output, label_var)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                end_time = time.time()
                
                # Record mIoU and pixel-wise accuracy
                pix_acc = cal_pixel_acc(output, label_var)
                mIoU = cal_mIoU(output, label_var)[-1]
                mIoU = torch.mean(mIoU)

                # Update recorders
                self.train_time.update(end_time - start_time)
                self.train_loss.update(loss.item())
                self.train_pix_acc.update(pix_acc.item())
                self.train_mIoU.update(mIoU.item())

                if (i + 1) % self.cfg.log_step == 0:
                    print('Epoch[{0}][{1}/{2}]\t'
                          'Time {train_time.val:.3f} ({train_time.avg:.3f})\t'
                          'Loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t'
                          'Pixel-Acc {train_pix_acc.val:.4f} ({train_pix_acc.avg:.4f})\t'
                          'mIoU {train_mIoU.val:.4f} ({train_mIoU.avg:.4f})'.format(
                              epoch + 1, i + 1, iter_per_epoch, 
                              train_time=self.train_time, 
                              train_loss=self.train_loss,
                              train_pix_acc=self.train_pix_acc,
                              train_mIoU=self.train_mIoU))
                
                #FIXME currently test validation code
                #if (epoch + 1) % self.cfg.val_step == 0:
                #    self.validate(epoch)
    
    def validate(self, epoch):
        """ Validate with validation dataset """
        self.model.eval()

        self.val_time.reset()
        self.val_loss.reset()
        self.val_mIoU.reset()
        self.val_pix_acc.reset()

        iter_per_epoch = len(self.val_loader) // self.cfg.val_batch_size
        if len(self.val_loader) % self.cfg.val_batch_size != 0:
            iter_per_epoch += 1

        for i, (image, label) in enumerate(self.val_loader):
                
            start_time = time.time()
            image_var = image.to(self.device)
            label_var = label.to(self.device)
            
            output = self.model(image_var)
            loss = self.criterion(output, label_var)

            end_time = time.time()
            
            # Record mIoU and pixel-wise accuracy
            pix_acc = cal_pixel_acc(output, label_var)
            mIoU = cal_mIoU(output, label_var)[-1]
            mIoU = torch.mean(mIoU)

            # Update recorders
            self.val_time.update(end_time - start_time)
            self.val_loss.update(loss.item())
            self.val_pix_acc.update(pix_acc.item())
            self.val_mIoU.update(mIoU.item())
            
            if (i + 1) % self.cfg.log_step == 0:
                print(' ##### Validation\t'
                      'Epoch[{0}][{1}/{2}]\t'
                      'Time {val_time.val:.3f} ({val_time.avg:.3f})\t'
                      'Loss {val_loss.val:.4f} ({val_loss.avg:.4f})\t'
                      'Pixel-Acc {val_pix_acc.val:.4f} ({val_pix_acc.avg:.4f})\t'
                      'mIoU {val_mIoU.val:.4f} ({val_mIoU.avg:.4f})'.format(
                          epoch + 1, i + 1, iter_per_epoch, 
                          val_time=self.val_time, 
                          val_loss=self.val_loss,
                          val_pix_acc=self.val_pix_acc,
                          val_mIoU=self.val_mIoU))
            
        if (epoch + 1) % self.cfg.sample_save_epoch == 0:
            pred = torch.argmax(output, dim=1)
            save_image(image, './sample/ori_' + str(epoch + 1) + '.png')
            save_image(label, './sample/true_' + str(epoch + 1) + '.png')
            save_image(pred, './sample/pred_' + str(epoch + 1) + '.png')

        if (epoch + 1) % self.cfg.model_save_epoch == 0:
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optim': self.optim.state_dict()
            }
            torch.save(state, './model/model_' + str(epoch + 1) + 'pth')
    
        #TODO:
        #   i) saving model
        #   ii) tensorboard or visdom
    
    def build_model(self):
        """ Rough """
        self.model = UNet(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.cfg.lr,
                                      betas=(self.cfg.beta0, self.cfg.beta1))
        if self.n_gpus > 1:
            print('### {} of gpus are used!!!'.format(self.n_gpus))
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        

    def build_recorder(self):
        self.train_time = AverageMeter()
        self.train_loss = AverageMeter()
        self.train_mIoU = AverageMeter()
        self.train_pix_acc = AverageMeter()

        self.val_time = AverageMeter()
        self.val_loss = AverageMeter()
        self.val_mIoU = AverageMeter()
        self.val_pix_acc = AverageMeter()

        # self.logger = Logger('./logs')

    
    def load_pre_model(self):
        """ Load pretrained model """
        print('=> loading checkpoint {}'.format(self.cfg.pre_model))
        checkpoint = torch.load(self.cfg.pre_model)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optim.load_state_dict(checkpoint['optim'])
        print('=> loaded checkpoint {}(epoch {})'.format(
            self.cfg.pre_model, self.start_epoch))

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
