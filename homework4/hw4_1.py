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
#%% CI-PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#ok so on some of the data we will be using train for training data
#and on some of the other data we will be using /train+/dev
#so it will be worth setting the train and eval data first for CI-PCA
files_train = glob.glob('*train.csv')
files_test = glob.glob('*eval.csv')
d_test = {}
for file in files_test:
    d_test["{0}".format(file)] = pd.read_csv(file,header=None)
    #define the labels and define the values of the vectors
    d_test["labels_{}".format(file)] = d_test[file].iloc[:,0]
    d_test["vectors_{}".format(file)] = d_test[file].iloc[:,1:3]  
    d_test.pop(file)

d_train = {}
for file in files_train:
    #all three dictionaries that have relevant training data
    #not going to worry about labels just yet
    d_train["{0}".format(file)]=pd.read_csv(file,header=None)
    #should have the x&y
    d_train["labels_{}".format(file)] = d_train[file].iloc[:,0]
    d_train["vectors_{}".format(file)] = d_train[file].iloc[:,1:3] 
    d_train.pop(file)
    
#x_train will always be the vectors
#y_train will always be the labels
sc = StandardScaler()
pca = PCA(n_components=(2))

x_train_keys = []
x_train_keys.append(list(d_train.keys())[1])
x_train_keys.append(list(d_train.keys())[3])
x_train_keys.append(list(d_train.keys())[5])
#print(x_train_keys) #these are the vectors

x_test_keys = []
x_test_keys.append(list(d_test.keys())[1])
x_test_keys.append(list(d_test.keys())[3])
x_test_keys.append(list(d_test.keys())[5])
#print(x_test_keys) #these are the vectors

#so now we have all the labels
for name in x_train_keys:
    sc.fit_transform(d_train[name])
    pca.fit(d_train[name])
    
for name in x_test_keys:
    sc.transform(d_test[name])
    pca.transform(d_test[name])
    
explained_variance = pca.explained_variance_ratio_

#now we should fit the model

classifier = LogisticRegression(random_state=(0))

#we need to call the labels now
#but we dont have them as a list
y_train_labels = []
y_train_labels.append(list(d_train.keys())[0])
y_train_labels.append(list(d_train.keys())[2])
y_train_labels.append(list(d_train.keys())[4])
#print(y_train_labels)

# class_data_keys = list(d_train.keys())
# class_data_val = list(d_train.values())







