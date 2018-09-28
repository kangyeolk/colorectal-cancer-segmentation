import os
import glob
import argparse
import random
import numpy as np

from utils import mkdir, move_files

def preprocess(config):

    root = config.root
    ratio = config.ratio
    # Remove key.txt
    if glob.glob(os.path.join(root, '*.txt')):
        os.remove(glob.glob(os.path.join(root, '*.txt'))[0])

    # Delete background parts
    file_list = os.listdir(root)
    png_names = [f[:-11] for f in file_list if 'png' in f]
    bg_names = [f for f in file_list if ('jpg' in f) and (f[:-4] not in png_names)]
    # assert len(bg_names) == len(png_names), 'Error'
    
    for bn in bg_names:
        os.remove(os.path.join(root, bn))
    
    # Split data by 3 categories: train, val, test
    train_ratio = ratio[0] / sum(ratio)
    val_ratio = ratio[1] / sum(ratio)
    image_files = sorted([f for f in os.listdir(root) if f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(root) if f.endswith('.png')])
    assert len(image_files) == len(label_files), 'The number of pairs is mismatched'

    n = len(image_files)
    idx = list(range(n))
    train_idx = np.random.choice(idx, size=int(n * train_ratio), replace=False)
    idx = [i for i in idx if i not in train_idx] 
    val_idx = np.random.choice(idx, size=int(n * val_ratio), replace=False)
    test_idx = [i for i in idx if i not in val_idx]

    train_images = [image_files[i] for i in train_idx]
    train_labels = [label_files[i] for i in train_idx]
    val_images = [image_files[i] for i in val_idx]
    val_labels = [label_files[i] for i in val_idx]
    test_images = [image_files[i] for i in test_idx]
    test_labels = [label_files[i] for i in test_idx]

    print('Train Images: {0}, Train Labels: {1}\t'
          'Validation Images: {2}, Validation Labels: {3}\t'
          'Test Images: {4}, Test Labels: {5}'.format(
              len(train_images), len(train_labels),
              len(val_images), len(val_labels),
              len(test_images), len(test_labels)))

    # Make Folders for saving
    mkdir(os.path.join(root, 'train'))
    mkdir(os.path.join(root, 'train', 'image'))
    mkdir(os.path.join(root, 'train', 'label'))
    
    mkdir(os.path.join(root, 'val'))
    mkdir(os.path.join(root, 'val', 'image'))
    mkdir(os.path.join(root, 'val', 'label'))
    
    mkdir(os.path.join(root, 'test'))
    mkdir(os.path.join(root, 'test', 'image'))
    mkdir(os.path.join(root, 'test', 'label'))


    # Move images, labels to directory
    move_files(root, root + '/train/image', train_images)
    move_files(root, root + '/train/label', train_labels)
    move_files(root, root + '/val/image', val_images)
    move_files(root, root + '/val/label', val_labels)
    move_files(root, root + '/test/image', test_images)
    move_files(root, root + '/test/label', test_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--ratio', type=int, default=[7,2,1],
                        nargs=3, metavar=('train', 'val', 'test'), help='Ratio of train, validation, test')
    
    config = parser.parse_args()
    preprocess(config)

    # takes 18 seconds for about 30,000 images