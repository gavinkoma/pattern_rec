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

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework4/data')
print(os.getcwd())

files = glob.glob('*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 

name_dict = {
    0: "data8dev.csv",
    1: "data8eval.csv",
    2: "data8train.csv",
    3: "data8train.csv",
    4: "data9dev.csv",
    5: "data9eval.csv",
    6: "data9train.csv",
    7: "data10dev.csv",
    8: "data10eval.csv",
    9: "data10train.csv"
}

#%% plots

data_val = list(d_all.values())
count = 0
for kval in data_val:
    plt.figure()
    #print(kval[1],kval[2])
    plt.scatter(kval[1],kval[2],c=kval[0],alpha=0.2)
    plt.title(str(count))
    plt.ylabel("y_vector")
    plt.xlabel("x_vector")
    plt.show()
    count += 1






