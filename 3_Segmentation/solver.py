import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import torch.backends.cudnn as cudnn

import time

from model import BinaryClassifier, UNet
from utils import cal_acc, cal_mIoU, cal_pixel_acc, denorm
# from logger import Logger


cudnn.benchmark = True

class Solver:

    def __init__(self, config, train_loader=None, val_loader=None, test_loader=None):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpus = self.cfg.n_gpus

        if self.cfg.mode in ['train', 'test']:
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

        # Trigger Tensorboard Logger
        if self.cfg.use_tensorboard:
            try:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter()
            except ImportError:
                print('=> There is no module named tensorboardX, tensorboard disabled')
                self.cfg.use_tensorboard = False

    
    def train_val(self):
        # Build record objs
        self.build_recorder()
        
        iter_per_epoch = len(self.train_loader.dataset) // self.cfg.train_batch_size
        if len(self.train_loader.dataset) % self.cfg.train_batch_size != 0:
            iter_per_epoch += 1
        
        for epoch in range(self.start_epoch, self.start_epoch + self.cfg.n_epochs):
            
            self.model.train()
            
            self.train_time.reset()
            self.train_loss.reset()
            self.train_cls_acc.reset()
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
                
                self.train_time.update(end_time - start_time)
                self.train_loss.update(loss.item())

                if self.cfg.task == 'cls':
                    # Record classification accuracy
                    cls_acc = cal_acc(output, label_var)

                    # Update recorder
                    self.train_cls_acc.update(cls_acc.item())

                    if (i + 1) % self.cfg.log_step == 0:
                        print('Epoch[{0}][{1}/{2}]\t'
                            'Time {train_time.val:.3f} ({train_time.avg:.3f})\t'
                            'Loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t'
                            'Accuracy {train_cls_acc.val:.4f} ({train_cls_acc.avg:.4f})'.format(
                                epoch + 1, i + 1, iter_per_epoch, 
                                train_time=self.train_time, 
                                train_loss=self.train_loss,
                                train_cls_acc=self.train_cls_acc))

                    if self.cfg.use_tensorboard:
                        self.writer.add_scalar('train/loss', loss.item(), epoch*iter_per_epoch + i)
                        self.writer.add_scalar('train/accuracy', cls_acc.item(), epoch*iter_per_epoch + i)

                
                elif self.cfg.task == 'seg':
                    # Record mIoU and pixel-wise accuracy
                    pix_acc = cal_pixel_acc(output, label_var)
                    mIoU = cal_mIoU(output, label_var)[-1]
                    mIoU = torch.mean(mIoU)

                    # Update recorders
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

                    if self.cfg.use_tensorboard:
                        self.writer.add_scalar('train/loss', loss.item(), epoch*iter_per_epoch + i)
                        self.writer.add_scalar('train/pix_acc', pix_acc.item(), epoch*iter_per_epoch + i)
                        self.writer.add_scalar('train/mIoU', mIoU.item(), epoch*iter_per_epoch + i)

                    
                #FIXME currently test validation code
                #if (i + 1) % 100 == 0:
            if (epoch + 1) % self.cfg.val_step == 0:
                self.validate(epoch)
        
        # Close logging
        self.writer.close()
    
    def validate(self, epoch):
        """ Validate with validation dataset """
        self.model.eval()

        self.val_time.reset()
        self.val_loss.reset()
        self.val_cls_acc.reset()
        self.val_mIoU.reset()
        self.val_pix_acc.reset()

        iter_per_epoch = len(self.val_loader.dataset) // self.cfg.val_batch_size
        if len(self.val_loader.dataset) % self.cfg.val_batch_size != 0:
            iter_per_epoch += 1

        for i, (image, label) in enumerate(self.val_loader):
                
            start_time = time.time()
            image_var = image.to(self.device)
            label_var = label.to(self.device)
            
            output = self.model(image_var)
            loss = self.criterion(output, label_var)

            end_time = time.time()

            self.val_time.update(end_time - start_time)
            self.val_loss.update(loss.item())
            
            if self.cfg.task == 'cls':
                # Record classification accuracy
                cls_acc = cal_acc(output, label_var)
                
                # Update recorder
                self.val_cls_acc.update(cls_acc.item())
                
                if (i + 1) % self.cfg.log_step == 0:
                    print('Epoch[{0}][{1}/{2}]\t'
                        'Time {val_time.val:.3f} ({val_time.avg:.3f})\t'
                        'Loss {val_loss.val:.4f} ({val_loss.avg:.4f})\t'
                        'Accuracy {val_cls_acc.val:.4f} ({val_cls_acc.avg:.4f})'.format(
                            epoch + 1, i + 1, iter_per_epoch, 
                            val_time=self.val_time, 
                            val_loss=self.val_loss,
                            val_cls_acc=self.val_cls_acc))

                if self.cfg.use_tensorboard:
                    self.writer.add_scalar('val/loss', loss.item(), epoch*iter_per_epoch + i)
                    self.writer.add_scalar('val/accuracy', cls_acc.item(), epoch*iter_per_epoch + i)
         
            elif self.cfg.task == 'seg':
                # Record mIoU and pixel-wise accuracy
                pix_acc = cal_pixel_acc(output, label_var)
                mIoU = cal_mIoU(output, label_var)[-1]
                mIoU = torch.mean(mIoU)

                # Update recorders
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

                if self.cfg.use_tensorboard:
                    self.writer.add_scalar('val/loss', loss.item(), epoch*iter_per_epoch + i)
                    self.writer.add_scalar('val/pix_acc', pix_acc.item(), epoch*iter_per_epoch + i)
                    self.writer.add_scalar('val/mIoU', mIoU.item(), epoch*iter_per_epoch + i)


        if self.cfg.task == 'cls':
            if (epoch + 1) % self.cfg.model_save_epoch == 0:
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optim': self.optim.state_dict()
                }
                if self.best_cls < self.val_cls_acc.avg:
                    self.best_cls = self.val_cls_acc.avg
                    torch.save(state, './model/cls_model_' + str(epoch + 1) + '_' + str(self.val_cls_acc.avg)[0:5] + '.pth')


        elif self.cfg.task == 'seg':
            # Save segmentation samples and model        
            if (epoch + 1) % self.cfg.sample_save_epoch == 0:
                pred = torch.argmax(output, dim=1)
                save_image(image, './sample/ori_' + str(epoch + 1) + '.png')
                save_image(label.unsqueeze(1), './sample/true_' + str(epoch + 1) + '.png')
                save_image(pred.cpu().unsqueeze(1), './sample/pred_' + str(epoch + 1) + '.png')

            if (epoch + 1) % self.cfg.model_save_epoch == 0:
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optim': self.optim.state_dict()
                }
                if self.best_seg < self.val_pix_acc.avg:
                    self.best_seg = self.val_pix_acc.avg
                    torch.save(state, './model/seg_model_' + str(epoch + 1) + '_' + str(self.val_pix_acc.avg)[0:5] + '.pth')
            
            if self.cfg.use_tensorboard:
                image = make_grid(image)
                label = make_grid(label.unsqueeze(1))
                pred = make_grid(pred.cpu().unqueeze(1))
                self.writer.add_image('Origianl', image, epoch + 1)
                self.writer.add_image('Labels', label, epoch + 1)
                self.writer.add_image('Predictions', pred, epoch + 1)
    
    def build_model(self):
        """ Rough """
        if self.cfg.task == 'cls':
            self.model = BinaryClassifier(num_classes=2)
        elif self.cfg.task == 'seg':
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
        # Train recorder
        self.train_time = AverageMeter()
        self.train_loss = AverageMeter()
        
        # For classification
        self.train_cls_acc = AverageMeter()
        # For segmentation
        self.train_mIoU = AverageMeter()
        self.train_pix_acc = AverageMeter()

        # Validation recorder
        self.val_time = AverageMeter()
        self.val_loss = AverageMeter()

        # For classification
        self.val_cls_acc = AverageMeter()
        # For segmentation
        self.val_mIoU = AverageMeter()
        self.val_pix_acc = AverageMeter()

        # self.logger = Logger('./logs')
        self.best_cls = 0
        self.best_seg = 0
    
    def load_pre_model(self):
        """ Load pretrained model """
        print('=> loading checkpoint {}'.format(self.cfg.pre_model))
        checkpoint = torch.load(self.cfg.pre_model)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optim.load_state_dict(checkpoint['optim'])
        print('=> loaded checkpoint {}(epoch {})'.format(
            self.cfg.pre_model, self.start_epoch))

    #TODO:Inference part:
    def infer(self, data):
        """
        input
            @data: iterable 256 x 256 patches
        output
            @output : segmentation results from each patch
                    i) If classifier's result is that there is a tissue inside of patch, outcome is a masked result.
                    ii) Otherwise, output is segmentated mask which all of pixels are background
        """ 
        # Data Loading

        # Load models of classification and segmetation and freeze them
        self.freeze()
        
        # Forward images to Classification model / Select targeted images

        # Forward images to Segmentation model

        # Record Loss / Accuracy / Pixel-Accuracy

        # Print samples out..

    def freeze(self):
        pass
        print('{}, {} have frozen!!!'.format('model_name_1', 'model_name_2'))


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
