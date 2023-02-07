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

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework4/data')
print(os.getcwd())

#%%begin by reading in data

files = glob.glob('*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 


#%%doesnt say custom so lets use sklearn!
#we need PCA, QDA, multi-class LDA --> compare to KNN & RNF
#note that the scales /all/ need to be the same

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#lets write the CI-pca one first
#so we need to first read through the training data
#so we should really just look for the train*

train = glob.glob('*train.csv')

d_train = {} #might as well keep the code objective with keys
for file in train:
    d_train["{}".format(file)] = pd.read_csv(file,header=None)

x1 = d_train['data8train.csv']
x1.rename(columns = {1:'x', 
                       2:'y'},
          inplace=True)
x1.drop(columns = [0])












