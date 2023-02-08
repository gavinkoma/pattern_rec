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

files = glob.glob('*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 


#%%
# #%%doesnt say custom so lets use sklearn!
# #we need PCA, QDA, multi-class LDA --> compare to KNN & RNF
# #note that the scales /all/ need to be the same

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression

# #lets write the CI-pca one first
# #so we need to first read through the training data
# #so we should really just look for the train*
# #there are three label options (0,1,2)
# #we should loop through the train values

# sc = StandardScaler()
# train = glob.glob('*train.csv')
# d_train = {}
# for file in train:
#     #might as well keep the code objective with keys
#     d_train["{}".format(file)] = pd.read_csv(file,delimiter=',',
#                                              header=None)

# #split the values as needed
# #your y train is 2D, dont forget this
# for file in train:
#     #should have the labels
#     d_train["train_labels_y_{}".format(file)] = d_train[file].iloc[:,0]
#     #should have the x&y
#     d_train["train_vectors_y{}".format(file)] = d_train[file].iloc[:,1:3] 
    

# trandict = {}
# for file in train:
#     d_train["yval_{}".format(file)] = sc.fit_transform(
#         d_train['{}'.format(file)])
#     trandict["labels_{}".format(file)] = d_train["train_labels_y_{}".format(file)]
    

# # pca = PCA(n_components=2)

# # for file in train:
# #     trandict["{}".format(file)] = pca.fit_transform(
# #         trandict["{}".format(file)])
    
# # explained_variance = pca.explained_variance_ratio_


#%% i guess we can plot first, im not sure what dev is

import matplotlib.pyplot as plt

data_keys = list(d_all.keys())
data_val = list(d_all.values())
count=0
for kval in data_val:
    plt.figure()
    #print(kval[1],kval[2])
    plt.scatter(kval[1],kval[2],c=kval[0],alpha=0.2)
    plt.title(str(data_keys[0]))
    plt.ylabel("y_vector")
    plt.xlabel("x_vector")
    plt.show()
    count+=1
    asdfasdfasdf

     





    
