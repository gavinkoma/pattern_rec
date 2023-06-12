#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:50:27 2023

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

#%% get 5d data
os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework6/5d')
print(os.getcwd())

filetrain = np.loadtxt('train.txt',dtype=float)
y_train = pd.DataFrame(filetrain[:,0])
x_train = pd.DataFrame(filetrain[:,1:6])

fileeval = np.loadtxt('eval_anonymized.txt',dtype=float)
y_eval = pd.DataFrame(fileeval[:,0])
x_eval = pd.DataFrame(fileeval[:,1:6])

name_dict = {0:'train_5d',
             1:'eval_5d'
             }

#cant really plot this? so i think im just gonna keep going

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



#%%
#gmm = GaussianMixture(n_components=val,random_state=0)
from sklearn.mixture import GaussianMixture as GMM


n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(x_train) for n in n_components]

plt.plot(n_components, [m.aic(x_train) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')








