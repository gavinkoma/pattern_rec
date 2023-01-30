#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:51:24 2023

@author: gavinkoma
"""
#import the basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read in the data
df_train = pd.read_csv('train.csv',
                       header=4,
                       names=["animal","xvec","yvec"])

df_eval = pd.read_csv('eval.csv',
                      header=4,
                      names=["animal","xvec","yvec"])

x_train = df_train[:][['xvec','yvec']]
#y_train = df_train[:]['animal']
x_test = df_eval[:][['xvec','yvec']]
#y_test = df_eval[:]['animal']

#generalized theorem:
    # P(class|data) = (P(data|class)*P(class))/P(data)
#we should first binary encode the aimals
#once again, we are going to use cats as 0 and dogs as 1
df_train.animal=[1 if i =="dogs" else 0 for i in df_train.animal]
df_eval.animal=[1 if i =="dogs" else 0 for i in df_eval.animal]
y_train = df_train[:]['animal']
y_test = df_eval[:]['animal']

#we dont need to split the data because it was previously prepared for us
#we have to also assume gaussian distribution
#two parameters mu and sigma^2 --> estimate from samples
mean = x_train.mean().transpose() #estimate the mean of the data
var = x_train.var() #estimate the variance for the data
probs = (0.5,0.5) #assume same priors
classes = np.unique(df_train["animal"].tolist()) #save all possible classes


#assume independence!!!!!!j















