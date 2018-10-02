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

    # Write Background parts
    file_list = os.listdir(root)
    png_names = [f[:-11] for f in file_list if 'png' in f]
    not_bg_names = [f for f in file_list if ('jpg' in f) and (f[:-4] in png_names)]
    bg_names = [f for f in file_list if ('jpg' in f) and (f[:-4] not in png_names)]
    assert len(not_bg_names) == len(png_names), \
           'The number of pairs is mismatched'
    

    #for bn in bg_names:
    #    os.remove(os.path.join(root, bn))
    
    # Split data by 3 categories: train, val, test
    train_ratio = ratio[0] / sum(ratio)
    val_ratio = ratio[1] / sum(ratio)
    
    bg_names = sorted(bg_names)
    not_bg_names = sorted(not_bg_names)
    label_names = sorted([f for f in file_list if 'png' in f])

    # Split paired data.
    n = len(not_bg_names)
    idx = list(range(n))
    train_idx = np.random.choice(idx, size=int(n * train_ratio), replace=False)
    idx = [i for i in idx if i not in train_idx] 
    val_idx = np.random.choice(idx, size=int(n * val_ratio), replace=False)
    test_idx = [i for i in idx if i not in val_idx]

    train_images = [not_bg_names[i] for i in train_idx]
    train_labels = [label_names[i] for i in train_idx]
    val_images = [not_bg_names[i] for i in val_idx]
    val_labels = [label_names[i] for i in val_idx]
    test_images = [not_bg_names[i] for i in test_idx]
    test_labels = [label_names[i] for i in test_idx]

    print('Train Images: {0}, Train Labels: {1}\t'
          'Validation Images: {2}, Validation Labels: {3}\t'
          'Test Images: {4}, Test Labels: {5}'.format(
              len(train_images), len(train_labels),
              len(val_images), len(val_labels),
              len(test_images), len(test_labels)))

    # Split background data
    n = len(bg_names)
    idx = list(range(n))
    train_idx = np.random.choice(idx, size=int(n * train_ratio), replace=False)
    idx = [i for i in idx if i not in train_idx] 
    val_idx = np.random.choice(idx, size=int(n * val_ratio), replace=False)
    test_idx = [i for i in idx if i not in val_idx]

    train_bg_images = [bg_names[i] for i in train_idx]
    val_bg_images = [bg_names[i] for i in val_idx]
    test_bg_images = [bg_names[i] for i in test_idx]

    print('Train background images: {}\t'
          'Validation background images: {}\t'
          'Test background images: {}'.format(
            len(train_bg_images), len(val_bg_images), len(test_bg_images)))
    

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

    # Add annotation information
    with open(os.path.join(root, 'train', 'annotation.txt'), 'w+') as f:
        for train_image in train_images:
            f.write('{}\t{}\n'.format(train_image, 1))
        for train_bg_image in train_bg_images:
            f.write('{}\t{}\n'.format(train_bg_image, 0))
    
    with open(os.path.join(root, 'val', 'annotation.txt'), 'w+') as f:
        for val_image in val_images:
            f.write('{}\t{}\n'.format(val_image, 1))
        for val_bg_image in val_bg_images:
            f.write('{}\t{}\n'.format(val_bg_image, 0))
    
    with open(os.path.join(root, 'test', 'annotation.txt'), 'w+') as f:
        for test_image in test_images:
            f.write('{}\t{}\n'.format(test_image, 1))
        for test_bg_image in test_bg_images:
            f.write('{}\t{}\n'.format(test_bg_image, 0))
    
    # Move images, labels to directory
    move_files(root, root + '/train/image', train_images)
    move_files(root, root + '/train/image', train_bg_images)
    move_files(root, root + '/train/label', train_labels)
    move_files(root, root + '/val/image', val_images)
    move_files(root, root + '/val/image', val_bg_images)
    move_files(root, root + '/val/label', val_labels)
    move_files(root, root + '/test/image', test_images)
    move_files(root, root + '/test/image', test_bg_images)
    move_files(root, root + '/test/label', test_labels)

## test..
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--ratio', type=int, default=[7,2,1],
                        nargs=3, metavar=('train', 'val', 'test'), help='Ratio of train, validation, test')
    
    config = parser.parse_args()
    preprocess(config)

    # takes 18 seconds for about 30,000 images