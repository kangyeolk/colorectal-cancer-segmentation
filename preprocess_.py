import konlpy
from konlpy.tag import Kkma
import re
import argparse
from utils import *
import time



def str2bool(x):
    return x.lower() in ['True', 'true', 1, 'y']

import pdb; pdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument('--pre_token', type=str2bool, default=False)
parser.add_argument('--comments', type=str, default=None)
parser.add_argument('--rm_list', type=str, default=None)
parser.add_argument('--text_file', type=str, default=None, help='input text file')
parser.add_argument('--vocab_file', type=str, default='vocab.txt', help='vocab file')
parser.add_argument('--vocab_max_size', type=int, default=10000, help='maximum vocabulary size')
parser.add_argument('--vocab_min_count', type=int, default=3, help='minimum frequency of the words')
args = parser.parse_args()

# ==================================== Make voca ========================================== #
# create vocabulary
if args.pre_token:
    print('create vocab')
    vocab = {}
    fp = open(args.text_file, 'r')
    for line in fp:
        arr = re.split('\s', line[:-1])
        for wd in arr:
            try:
                vocab[wd] += 1
            except:
                vocab[wd] = 1
    fp.close()
    vocab_arr = [[wd, vocab[wd]] for wd in vocab if vocab[wd] > args.vocab_min_count]
    vocab_arr = sorted(vocab_arr, key=lambda k: k[1])[::-1]
    vocab_arr = vocab_arr[:args.vocab_max_size]
    vocab_arr = sorted(vocab_arr)

    fout = open(args.vocab_file, 'w')
    for itm in vocab_arr:
        itm[1] = str(itm[1])
        fout.write(' '.join(itm)+'\n')
    fout.close()


# ==================================== Preprocess ========================================== #
def pre_all(text, rm_list, rep_list):
    for rm in rm_list:
        text = [t.replace(rm, '') for t in text]
    for abb, rep in rep_list.items():
        text = [t.replace(abb, rep) for t in text]
    return text

# Recall remove list
rm_list = []

#with open('rm_list.txt', 'r') as f:
#    rm_list.append(f.readlines())


# Adding point
add_point = True

with open('vocab.txt', 'r') as f:
    while(add_point):
        tmp = []
        for ii in range(5):
            word, _ = f.readline().split()
            tmp.append(word)
        ans = input('Add {0}, {1}, {2}, {3}, {4} to remove list?[y/n/number]'.format(
                    tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]))
        
        if ans.lower() == 'y':
            rm_list.append(tmp)
        elif ans.lower() == 'n':
            add_point = False
        else:
            num = int(ans)
            for jj in range(num):
                rm_list.append(tmp[jj])
            add_point = False
        
# Misc
rm_list +=  ['\n', ':', 'ㅇㅅㅇ', '~', '.',  '?', ',', '?', '!', ';',
           '♥', '♡', '*', '●', '□', 'x', '/', 'ㄴ', 'ㅂ', 'ㄹ', 'x', 'ㅅ', 'ㅎ' 
            'ㅋ', 'ㅉ', 'ㅠ', 'ㅡ', 'ㅈ', '❤️'
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '^^']

rep_list = {'ㅂㅅ':'병신', 'ㅅㅂ':'시발', 'ㅅㄲ':'새끼'}


# Preprocessing -> Tokenize -> Write
with open(args.comments, 'r', encoding='utf-8') as  f:
    line = f.readlines()
    line = pre_all(line, rm_list, rep_list)


start_time = time.time()
token = []
kkma = Kkma()
for l in line:
    token.append(kkma.pos(l))
print('Tokenization took about {} secs'.format(str(time.time() - start_time)))

with open(args.comments[:-4] + '_kkma.txt', 'w', encoding='utf-8') as f:
    for tok in token:
        for tt in tok:
            if len(tt[0]) != 1:
                f.write(tt[0] + ' ')
        f.write('\n')




