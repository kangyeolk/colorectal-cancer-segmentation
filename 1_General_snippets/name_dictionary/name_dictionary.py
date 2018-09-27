import re
import os
import pickle

def main():
    root = '/home/keetae/remote/share/Medical'
    svs_dir = root + '/data/pathology_images_copy'

    dir = make_svs_list(svs_dir)
    namedict = clean_name_save(dir) # clean names, save, and make namedict
    save_namedict(namedict)
    change_filenames(svs_dir, namedict)



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


def clean_name_save(dir, savefile="./namedict.csv",):
    '''
    clean names, save, and make namedict
    excel file saved in root/namedict.csv

    :param dir: list containing all the svs file names
    :return: dictionary containing link information
    '''
    namedict = {}
    p = 1
    f = open(savefile, 'w')
    f.write('orig_name, refined_name, new name\n')
    for name in dir:
        rename = re.sub('^(.*)(?=S)', "", name)
        rename = re.sub('\s\s\s', '_', rename)
        rename = re.sub(r'[\^]', '', rename)
        rename = re.sub(r'[\#]', '_', rename)[:-1]
        rename = re.sub('\s', '', rename)
        idx = 'P' + "%04d" % p

        namedict[name] = (rename, idx)
        print(name + ',' + rename + ',' + idx)
        f.write(name + ',' + rename + ',' + idx + '\n')
        p += 1
    print('saving in %s done!'%savefile)
    f.close()
    return namedict


def save_namedict(namedict, savefile='./namedict.pkl'):
    f = open(savefile, 'wb')
    pickle.dump(namedict, f)
    f.close()

def change_filenames(svs_dir, namedict):
    '''
    caution!! it replace all the file names, thus the only clue is the save excel fiel and the dictionary.
    :param svs_dir:
    :param namedict:
    :return: all filename changed as P0001, P0002, ~
    '''
    makesure = input('Will you change the file names? you better make a copy first [yes or no] :')
    print(makesure)
    while not makesure in ['yes', 'no']:
        input('Please indicate True or False')

    if makesure=='yes':
        for filename in os.listdir(svs_dir):
            os.rename(os.path.join(svs_dir,filename), os.path.join(svs_dir, namedict[filename[:-4]][1]+'.svs'))
        print('all file name changed')
    else:
        print('no name changed')

if __name__ == '__main__':
    main()
