#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:55:07 2023

@author: gavinkoma
"""
#okay so we know the skeleton of our naive bayes work
#1. we first have to get all of the summary statistics 
#2. get the probability of the class for each sample
#assume gaussian and independent!

#import the modules
import pandas as pd
import numpy as np
import math

#as always, load the data
#read in the data
df_train = pd.read_csv('train.csv',
                       header=4,
                       names=["animal","xvec","yvec"])
X,y = df_train

df_eval = pd.read_csv('eval.csv',
                      header=4,
                      names=["animal","xvec","yvec"])

x_train = df_train[:][['xvec','yvec']]
y_train = df_train[:]['animal']
x_test = df_eval[:][['xvec','yvec']]
y_test = df_eval[:]['animal']


class NiaveBayes:
    def fit(self,df_train,):
        pass