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

#%% plots
def plots():
    data_keys = list(d_all.keys())
    data_val = list(d_all.values())
    for kval in data_val:
        plt.figure()
        #print(kval[1],kval[2])
        plt.scatter(kval[1],kval[2],c=kval[0],alpha=0.2)
        plt.title(str(data_keys[0]))
        plt.ylabel("y_vector")
        plt.xlabel("x_vector")
        plt.show()
        
plots()






