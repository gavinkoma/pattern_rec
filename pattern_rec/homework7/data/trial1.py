#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:27:21 2023

@author: gavinkoma
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework7/data')
print(os.getcwd())

dev = pd.read_csv('dev.csv',header=None)
train = pd.read_csv('train.csv',header=None)

dev_label = dev.iloc[:,0]
dev_x1 = dev.iloc[:,1]
dev_x1 = dev.iloc[:,2]

train_label = train.iloc[:,0]
train_x1 = train.iloc[:,1]
train_x2 = train.iloc[:,2]

def plot(dev_data, eval_data):
    plt.figure()
    plt.scatter(dev[1],dev[2],c=dev[0],alpha=0.2)
    plt.title('dev_data')
    plt.show()
    
    plt.figure()
    plt.scatter(train[1],train[2],c=train[0],alpha=0.2)
    plt.title('eval_data')
    plt.show()
    return

def normalization(data):
    data_normalized = []
    x_min = data.min()
    x_max = data.max()
    
    for i in data:
        y=round(((i-x_min)/(x_max-x_min))*128)
        data_normalized.append(y)
        
    return data_normalized

x1dev = pd.DataFrame(normalization(dev.iloc[:,1]))
x2dev = pd.DataFrame(normalization(dev.iloc[:,2]))
x1train = pd.DataFrame(normalization(train.iloc[:,1]))
x2train = pd.DataFrame(normalization(train.iloc[:,2]))

se_x1dev = round(entropy(x1dev.value_counts(normalize=True)),4)
se_x2dev = round(entropy(x2dev.value_counts(normalize=True)),4)
se_x1train = round(entropy(x1train.value_counts(normalize=True)),4)
se_x2train = round(entropy(x2train.value_counts(normalize=True)),4)

##change to list for joint
x1dev = x1dev.values.tolist()
x2dev = x2dev.values.tolist()
x1train = x1train.values.tolist()   
x2train = x2train.values.tolist()

freq_dev = pd.crosstab(x1dev,x2dev)
freq_train = pd.crosstab(x1train,x2train)
joint_dev = round(entropy(freq_dev.values.flatten(),base=2),4)
joint_train = round(entropy(freq_train.values.flatten(),base=2),4)
print(joint_dev)
print(joint_train)

#generate the random dataset
num_rows = 15000
feature1 = np.random.normal(loc=0.0,scale=1.0,size=(num_rows,))
feature2 = np.random.normal(loc=0.0,scale=1.0,size=(num_rows,))
labels = np.random.choice([0,1,2],size=num_rows)

random = {'feature1':feature1,
          'feature2':feature2,
          'class':labels
          }

randomdata = pd.DataFrame(random)
feature1_normal = normalization(randomdata['feature1'])
feature2_normal = normalization(randomdata['feature2'])

freq_random = pd.crosstab(feature1_normal,feature2_normal)
joint_se = round(entropy(freq_random.values.flatten(),base=2),4)



# parent_dir = "/Users/gavinkoma/Desktop/pattern_rec/homework7/data"
# data_paths = ['train','dev']
# col_names = ['classes','feature1','feature2']

# df_dict={}












