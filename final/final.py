# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import dropwhile

os.chdir("/Users/gavinkoma/Desktop/pattern_rec/final/subset_data/")

name_list = []
dct = {}
for x in os.listdir():
    if x.endswith(".csv"):
        #print(x)
        name_list.append(x)

def is_comment(s):
    return s.startswith("#")

for name in name_list:
    dct['lst_%s' % name] = []
    with open(name,'r') as fh:
        for curline in dropwhile(is_comment,fh):
            dct['lst_%s' % name].append(curline)
           # dct['lst_%s' % name['number','values']] = dct['lst_%s' % name].str.split(',',expand=True)


# df[['number','value']] = df[0].str.split(',',expand=True)
# df.pop(0)


# plt.figure()
# plt.scatter(df['number'],df['value'])
# plt.show()

