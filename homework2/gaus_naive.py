#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:33:53 2023

@author: gavinkoma
"""

#%%import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% data load
#as always, load the data
#read in the data
df_train = pd.read_csv('train.csv',
                       header=4,
                       names=["animal","xvec","yvec"])
df_eval = pd.read_csv('eval.csv',
                      header=4,
                      names=["animal","xvec","yvec"])

df_train.animal=[1 if i =="dogs" else 0 for i in df_train.animal]
df_eval.animal=[1 if i =="dogs" else 0 for i in df_eval.animal]

#make it all as numpy for consistency
x_train = df_train[:][['xvec','yvec']]
dataset_train = np.array(x_train)
label_train = np.array(df_train[:]['animal'])

x_test = df_eval[:][['xvec','yvec']]
dataset_test = np.array(x_test)
label_test = np.array(df_eval[:]['animal'])

#%% gotta make the fit

X = dataset_train[:,0:2]
t = label_train[:]

priors = {}
means = {}
covs = {}
classes = np.unique(t)

for c in classes:
    X_c = X[t==c]
    priors = {0:0.5,1:0.5}
    means[c] = np.mean(X_c,axis=0)
    covs[c] = np.cov(X_c,rowvar=False)
        
def predict(X):
    preds = []
    for x in X:
        posts = []
        for c in classes:
            prior = np.log(priors[c])
            inv_cov = np.linalg.inv(covs[c])
            inv_cov_det = np.linalg.det(inv_cov)
            diff = x-means[c]
            likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
            post = prior + likelihood
            posts.append(post)
            
        pred = classes[np.argmax(posts)]
        preds.append(pred)
        
    return np.array(preds)

#%%run the training eval
preds = predict(X)
accuracy_score_train = sum(label_train == preds)/len(label_train)
print(accuracy_score_train)

#%%run the test eval
X = dataset_test[:,0:2]
t = label_test[:]
preds = predict(X)
accuracy_score_test = sum(label_test == preds)/len(label_test)
print(accuracy_score_test)

#%%alter the prior values with increments
#make the range for the data
prior_dog = np.arange(0.0,1.01,0.01)

prior_cat = []#make list for 1-p(dog)
for val in np.nditer(prior_dog):
    prior_cat.append(1-val)
    
prior_cat = np.array(prior_cat)
priorval = np.array((prior_cat,prior_dog)).T

X = dataset_test[:,0:2]
t = label_test[:]

priors = {}
means = {}
covs = {}
classes = np.unique(t)

def predict_range(X,cat,dog):
    
    for c in classes:
        X_c = X[t==c]
        priors = {0:cat,1:dog}
        means[c] = np.mean(X_c,axis=0)
        covs[c] = np.cov(X_c,rowvar=False)
    
    pred_ranges = []
    
    for x in X:
        posts = []
        for c in classes:
            prior = np.log(priors[c])
            inv_cov = np.linalg.inv(covs[c])
            inv_cov_det = np.linalg.det(inv_cov)
            diff = x-means[c]
            likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
            post = prior + likelihood
            posts.append(post)
            
        predrange = classes[np.argmax(posts)]
        pred_ranges.append(predrange)

    return np.array(pred_ranges)

errorvalues = []

for cat,dog in priorval:
    pred_ranges = predict_range(X,cat,dog)
    accuracy_score_range = sum(label_test == pred_ranges)/len(label_test)
    errorvalues.append(accuracy_score_range)
    print(cat,dog)
    
#%% show the plot
rate = []
for val in errorvalues:
    trueerror = 1-val
    rate.append(trueerror)

plt.show()
plt.title("Plot of Error Rate with Increasing Dog-Priors")
plt.ylabel("Error Rate")
plt.xlabel("Dog Prior")
plt.scatter(prior_cat,rate)
