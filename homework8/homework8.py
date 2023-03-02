#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:12:44 2023

@author: gavinkoma
"""

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework8/')
print(os.getcwd())

dev = pd.read_csv('eval.csv',header=None)
train = pd.read_csv('train.csv',header=None)
x1_dev = dev.iloc[:,1]
x2_dev = dev.iloc[:,2]
x1_train = train.iloc[:,1]
x2_train = train.iloc[:,2]

files = [dev,train]

#as always, lets visualize our data yeah?
def visualize(files):
    for file in files:
        plt.figure()
        plt.scatter(file.iloc[:,1],file.iloc[:,2],c=file.iloc[:,0],alpha=0.2)
        plt.show()    
        
#visualize(files)


#we should quantize these vectors?? 










