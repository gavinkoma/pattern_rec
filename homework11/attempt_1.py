#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:21:01 2023

@author: gavinkoma
"""


#%%import libraries
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


#%%import data and do very basic graphs
os.chdir("/Users/gavinkoma/Desktop/pattern_rec/homework11/data")
train_data = pd.read_csv("train_03.csv",header=None)
dev_data = pd.read_csv("dev_03.csv",header=None)
eval_data = pd.read_csv("eval_03.csv",header=None)

plt.figure()
sns.scatterplot(x=train_data.iloc[:,1],
                y=train_data.iloc[:,2],
                data=train_data,
                hue=train_data.iloc[:,0]).set_title("train_data")


plt.figure()
sns.scatterplot(x=dev_data.iloc[:,1],
                y=dev_data.iloc[:,2],
                data=dev_data,
                hue=dev_data.iloc[:,0]).set_title("dev_data")


plt.figure()
sns.scatterplot(x=eval_data.iloc[:,1],
                y=eval_data.iloc[:,2],
                data=eval_data,
                hue=eval_data.iloc[:,0]).set_title("eval_data")


#%%implementation of SLP








