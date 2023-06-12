#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:32:39 2023

@author: gavinkoma
"""

#%% start with modules
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework4/data')
print(os.getcwd())

files = glob.glob('*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 

name_dict = {
    0: "data10train",#data10train
    1: "data9train", #data9train
    2: "data8train", #data8train
    3: "data9eval", #data9eval
    4: "data8dev",#data8dev
    5: "data10dev",#data10dev
    6: "data10val",#data10eval
    7: "data8eval", #data8eval
    8: "data9dev" #data9dev
}

custom_xlim = (0, 100)
custom_ylim = (-100, 100)

#%% plots

data_val = list(d_all.values())

for i,kval in enumerate(data_val):
    plt.figure()
    #print(kval[1],kval[2])
    plt.scatter(kval[1],kval[2],c=kval[0],alpha=0.2)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.title(name_dict[i])
    plt.ylabel("y_vector")
    plt.xlabel("x_vector")
    plt.savefig(str(name_dict[i]))
    plt.show()



