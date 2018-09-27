import torch
import torch.nn as nn

import argparse
from utils import str2bool, mkdir
from data_loader import get_loader
from solver import Solver

def main(config):

    if config.sample_save:
        mkdir(config.sample_save_path)
    if config.model_save:
        mkdir(config.sample_save_path)
    
    if config.mode == 'train':
        # Get Dataloder
        train_data_loader = get_loader(config=config,
                                       data_type='train')
        val_data_loader = get_loader(config=config,
                                     data_type='val')
        
        # Training
        solver = Solver(config=config,
                        train_loader=train_loader,
                        val_loader=val_loader)
        solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training setting
    parser.add_argument('--mode', type=str, default=train, choice=['train'])
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--beta0', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.99)

    # Path
    parser.add_argument('--train_data_path', type=str, default='./train')
    parser.add_argument('--val_data_path', type=str, default='./val')
    parser.add_argument('--test_data_path', type=str, default='./test')
    parser.add_argument('--sample_save_path', type=str, default='./samples')
    parser.add_argument('--model_save_path', type=str, './model')

    # Logging
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--sample_save', type=str2bool, default='True')
    parser.add_argument('--model_save', type=str2bool, default='True')
    parser.add_argument('--use_visdom', type=str2bool, default='True')

    # Misc
    parser.add_argument('--pre_model', type=str, default=None)

    config = parser.parse_args()
    print(config)
    main(config)



