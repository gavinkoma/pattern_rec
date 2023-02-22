#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:15:57 2023

@author: gavinkoma
"""
#%%import and chdir
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework6/')
print(os.getcwd())

#%%lets begin
#change directory to data (2d)
os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework6/2d/data')
print(os.getcwd())

#import the data bruv
filetrain = np.loadtxt("train.txt", dtype=float)
y_train = pd.DataFrame(filetrain[:,0])
x_train = pd.DataFrame(filetrain[:,1:3])
 
fileeval = np.loadtxt("eval.txt", dtype=float)
y_eval = pd.DataFrame(fileeval[:,0])
x_eval = pd.DataFrame(fileeval[:,1:3])

name_dict = {0:"Train",
             1:"Eval"
             }
 
def plot(file):
    plt.figure()
    #print(kval[1],kval[2])
    plt.scatter(file[:,1],file[:,2],c=file[:,0],alpha=0.2)
    plt.title(name_dict[count])
    
for count in [0]:
    plot(filetrain)
    for count in [1]:
        plot(fileeval)

#%%
#there are two different classes here 0,1 --> binary --> love!
#why dont we plot first to get an idea of stuff 

def GM_plot(x,y):
    plt.figure()
    d = x
    gmm.fit(d)
    labels = gmm.predict(d)
    d['labels'] = labels
    d0 = d[d['labels'] == 0]
    d1 = d[d['labels'] == 1]
    plt.scatter(d0[0],d0[1],c='r')
    plt.scatter(d1[0],d1[1],c='g')
    plt.title(name_dict[i])
    plt.show()

name_dict = {
    0: "traindata",#data10train
    1: "evaldata", #data9train
    2: "traindata", #data8train
    3: "evaldata", #data9eval
    }

for i,val in enumerate([1,2,4,8]):
    gmm = GaussianMixture(n_components=val,random_state=0)
    GM_plot(x_train,y_train)

for i,val in enumerate([1,2,4,8]):
    gmm = GaussianMixture(n_components=val,random_state=0)
    GM_plot(x_eval,y_eval)

