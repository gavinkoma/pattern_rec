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
    0: "data8dev",
    1: "data8eval",
    2: "data8train",
    3: "data8train",
    4: "data9dev",
    5: "data9eval",
    6: "data9train",
    7: "data10dev",
    8: "data10eval",
    9: "data10train"
}

#%% plots

data_val = list(d_all.values())
for i,kval in enumerate(data_val):
    #print(kval[1],kval[2])
    plt.scatter(kval[1],kval[2],c=kval[0],alpha=0.2)
    plt.title(name_dict[i])
    plt.ylabel("y_vector")
    plt.xlabel("x_vector")
    plt.savefig(str(name_dict[i]))
    plt.show()



