- This program changes the original messy name such as

1002627_S 180028742#1   2#

into

S180028742_1_2

and assign a patient number as

P0075.

- The triple (1002627_S 180028742#1   2#, S180028742_1_2, P0075) are saved in an excel file
named ./namedict.csv

- The dictionary, ex) namedict['1002627_S 180028742#1   2#'] = (S180028742_1_2, P0075)
will be saved in ./namedict.pkl

- running this file will prompt whether to change the original file name.