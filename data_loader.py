import os
import numpy as np
from PIL import Image
import scipy.misc


if __name__ == '__main__':
   img = Image.open('./WQU_dataset/train/train_2_anno.bmp').convert('RGB')
   img = np.array(img)
   for i in range(7):
        img[np.where(img == i+1)] = 30 * i
   # 각각의 gland마다 색이 조금씩 다름 통일하고 하셈..
   #img[np.where(img == 1)] = 255
   #img[np.where(img == 3)] = 125
   # _img = img.reshape(img.shape[0]*img.shape[1]*3)
   # print(np.unique(_img[np.where(_img != 0)]))
   scipy.misc.imsave('outfile2.jpg', img)