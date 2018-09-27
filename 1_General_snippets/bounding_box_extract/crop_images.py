import os
import cv2
import openslide
import scipy.misc
import numpy as np



def main():
    bounding_box_dict = {}  # save bounding boxes
    root = '/home/keetae/remote/share/Medical/'
    svs_dir = root + '/data/pathology_images2'
    save_dir = root + 'save2'

    f_list = make_svs_list(svs_dir) # make working list

    f_list=filter_svs_list(save_dir, f_list)


    crop_and_save(f_list, bounding_box_dict, save_dir, svs_dir) # do all the tricks



def filter_svs_list(save_dir, filename_list):
    saved_list = []
    for filename in os.listdir(save_dir):
        saved_list.append(filename)
    new_filename_list = []
    for curr_filename in filename_list:
        if curr_filename in saved_list:
            print('%s already in the list!'%curr_filename)
            continue
        else:
            new_filename_list.append(curr_filename)
    print('new file length : %d'%(len(new_filename_list)))
    return new_filename_list

def make_svs_list(svs_dir):
    '''
    :param svs_dir: svs image data directory
    :return: list of file names without .svs extension
    '''
    filename_list = []
    for filename in os.listdir(svs_dir):
        filename_list.append(filename[:-4])
    print('total number of svs file is: %d' % (len(filename_list)))
    return filename_list


def crop_and_save(f_list, bounding_box_dict, save_dir, svs_dir):
    '''
    :param f_list: list of file names without extension
    :param bounding_box_dict: dictionary for saving the bounding box
    :param save_dir: location to save box images
    :param svs_dir: svs data location
    :return: bounding box dictionary

    This function implements all later functions one file for each time
    '''
    for i in range(len(f_list)):
        sample_name = f_list[i]
        make_folder(save_dir, sample_name)
        svs_img = openslide.OpenSlide(os.path.join(svs_dir, sample_name + '.svs'))
        w_img = make_thumb_img(svs_img)
        bounding_box_dict = get_bounding_box(w_img, bounding_box_dict, sample_name, safety=1800 // 20)
        save_cropped_box(svs_img, bounding_box_dict, sample_name, save_dir)
    return bounding_box_dict


def make_folder(path, version):
    '''
    :param path: root path
    :param version: name of the folder
    :return: folder made
    '''
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))


def make_thumb_img(img):
    '''
    input: one svs loaded image
    output: opencv object of thumbnail version of svs. smaller version for bounding box detection
    '''
    size1, size2 = img.dimensions
    if size1 > size2:
        th_dim = size1 / 20
    else:
        th_dim = size2 / 20
    img_thumb = img.get_thumbnail((th_dim, th_dim))
    img_thumb_np = np.array(img_thumb)
    scipy.misc.imsave('temp.jpg', img_thumb_np)
    working_img = cv2.imread('./temp.jpg')
    return working_img


def get_bounding_box(orig_img, bounding_box_dict, sample_name, safety=1800 // 20):
    blur = cv2.medianBlur(orig_img, 155)
    _, threshed_img = cv2.threshold(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY),
                                    230, 255, cv2.THRESH_BINARY)
    _, bound, _ = cv2.findContours(threshed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bounding_box_dict[sample_name] = {}
    bounding_box_dict[sample_name]['num'] = 0

    for i in range(len(bound) - 1):
        x, y, w, h = cv2.boundingRect(bound[i])
        bounding_box_dict[sample_name][i] = ((x - safety) * 20, (y - safety) * 20, (w + 2 * safety) * 20, (h + 2 * safety) * 20)
        bounding_box_dict[sample_name]['num'] += 1
    return bounding_box_dict


def save_cropped_box(orig_img, bounding_box_dict, sample_name, save_dir):
    '''
    :param orig_img: the svs image working on
    :param bounding_box_dict: dict.
    :param sample_name: current file name
    :param save_dir: where to save
    :return: saved in a single file
    '''
    for i in range(bounding_box_dict[sample_name]['num']):
        curr_box = bounding_box_dict[sample_name][i]
        try:
            image2save = orig_img.read_region((curr_box[0], curr_box[1]), 0, (curr_box[2], curr_box[3])).convert("RGB")
        except MemoryError:
            print('memoryerror occurred for %s'%sample_name)
            f = open("./error_log2.txt", 'a')
            f.write('sample %s, %d-th error\n '%(sample_name, i))
            f.close()
            continue

        image2save.save(os.path.join(save_dir, sample_name, '%d' % (i) + '.tiff'), compression='tiff_lzw')
        print('%dth saved for sample %s' % (i + 1, sample_name))
    print('%s extract finish!'%(sample_name))



if __name__ == '__main__':
    main()