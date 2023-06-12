#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:26:19 2023

@author: gavinkoma
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedStratifiedKFold


os.chdir("/Users/gavinkoma/Desktop/pattern_rec/homework9/data")

#lets just assume our chosen system is a decision tree
#%% dataprep
#use the entire training data set
conventional_train = pd.read_csv("00_train_03.csv",header=None)
conventional_dev = pd.read_csv("00_dev_03.csv",header=None)
conventional_eval = pd.read_csv("00_eval_03.csv",header=None)

#before using pca we whould train the model and use a logistic
#regression to just see how well it performs
x_train = conventional_train.iloc[:,1:3]
y_train = conventional_train.iloc[:,0]
x_dev = conventional_dev.iloc[:,1:3]
y_dev = conventional_dev.iloc[:,0]
x_eval = conventional_eval.iloc[:,1:3]
y_eval = conventional_eval.iloc[:,0]


#for bag we need random 75% of dataset
conventional_train_shuffle = conventional_train.sample(frac=1)
conv_train_75 = conventional_train_shuffle.iloc[0:75000,:]
x_75_train = conv_train_75.iloc[:,1:3]
y_75_train = conv_train_75.iloc[:,0]

#bagging method
LR = LogisticRegression(random_state=0)

bag = BaggingClassifier(base_estimator=LR,
                        n_estimators=10,
                        random_state=0)

#this bag is 75% of the original dataset split into 10
bag.fit(x_75_train,y_75_train) 



#CV method
k_folds = KFold(n_splits=10)
scores = cross_val_score(LR, x_train, y_train, cv = k_folds)

#%%construct systems now
#%%(A) conventional --> baseline pca system
pca = PCA(n_components=(2))
pca.fit(x_train,y_train)

y_train_pca = y_train
x_train_pca = pca.fit_transform(x_train)

x_dev_pca = pca.fit_transform(x_dev)
x_eval_pca = pca.fit_transform(x_eval)

logreg = LogisticRegression()
logreg.fit(x_train_pca, y_train_pca)
ypred_train = logreg.predict(x_train_pca)
ypred_dev = logreg.predict(x_dev_pca)
ypred_eval = logreg.predict(x_eval_pca)

acc_log_train = logreg.score(x_train_pca, y_train_pca)
print(1-acc_log_train)

acc_log_dev = logreg.score(x_dev_pca,y_dev)
print(1-acc_log_dev)

acc_log_eval = logreg.score(x_eval_pca,y_eval)
print(1-acc_log_eval)

#done with pca

y_pred_proba = logreg.predict_proba(x_train)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_train, y_pred_proba)
plt.plot(fpr,tpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")


#%% (B) conventional bootstrapping

#use the entire training data set
conventional_train = pd.read_csv("00_train_03.csv",header=None)
conventional_dev = pd.read_csv("00_dev_03.csv",header=None)
conventional_eval = pd.read_csv("00_eval_03.csv",header=None)

#before using pca we whould train the model and use a logistic
#regression to just see how well it performs
x_train = conventional_train.iloc[:,1:3]
y_train = conventional_train.iloc[:,0]
x_dev = conventional_dev.iloc[:,1:3]
y_dev = conventional_dev.iloc[:,0]
x_eval = conventional_eval.iloc[:,1:3]
y_eval = conventional_eval.iloc[:,0]

# train model
reg = LogisticRegression(random_state=0)
reg.fit(x_train, y_train)
n_iterations = 1000

# bootstrap predictions
accuracy_train = []
for i in range(n_iterations):
    X_bs, y_bs = resample(x_train, y_train, replace=True)
    # make predictions
    y_hat = reg.predict(X_bs)
    # evaluate model
    score = accuracy_score(y_bs, y_hat)
    accuracy_train.append(score)
print(1-sum(accuracy_train)/len(accuracy_train))

accuracy_dev = []
for i in range(n_iterations):
    X_bs, y_bs = resample(x_train, y_train, replace=True)
    # make predictions
    y_hat = reg.predict(X_bs)
    # evaluate model
    score = accuracy_score(y_bs, y_hat)
    accuracy_dev.append(score)
print(1-sum(accuracy_dev)/len(accuracy_dev))

accuracy_eval = []
for i in range(n_iterations):
    X_bs, y_bs = resample(x_train, y_train, replace=True)
    # make predictions
    y_hat = reg.predict(X_bs)
    # evaluate model
    score = accuracy_score(y_bs, y_hat)
    accuracy_eval.append(score)
print(1-sum(accuracy_eval)/len(accuracy_eval))

#%% (C) conventional with cross validation
# evaluate a logistic regression model using k-fold cross-validation

#use the entire training data set
conventional_train = pd.read_csv("00_train_03.csv",header=None)
conventional_dev = pd.read_csv("00_dev_03.csv",header=None)
conventional_eval = pd.read_csv("00_eval_03.csv",header=None)

#before using pca we whould train the model and use a logistic
#regression to just see how well it performs
x_train = conventional_train.iloc[:,1:3]
y_train = conventional_train.iloc[:,0]
x_dev = conventional_dev.iloc[:,1:3]
y_dev = conventional_dev.iloc[:,0]
x_eval = conventional_eval.iloc[:,1:3]
y_eval = conventional_eval.iloc[:,0]


# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
model.fit(x_train,y_train)
# evaluate model
scores = cross_val_score(model, 
                         x_train, 
                         y_train, 
                         scoring='accuracy', 
                         cv=cv, 
                         n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

scores = cross_val_score(model, 
                         x_dev, 
                         y_eval, 
                         scoring='accuracy', 
                         cv=cv, 
                         n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


scores = cross_val_score(model, 
                         x_eval, 
                         y_eval, 
                         scoring='accuracy', 
                         cv=cv, 
                         n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


#%% (D) bootstrap periodt

from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#use the entire training data set
conventional_train = pd.read_csv("00_train_03.csv",header=None)
conventional_dev = pd.read_csv("00_dev_03.csv",header=None)
conventional_eval = pd.read_csv("00_eval_03.csv",header=None)

#before using pca we whould train the model and use a logistic
#regression to just see how well it performs
x_train = conventional_train.iloc[:,1:3]
y_train = conventional_train.iloc[:,0]
x_dev = conventional_dev.iloc[:,1:3]
y_dev = conventional_dev.iloc[:,0]
x_eval = conventional_eval.iloc[:,1:3]
y_eval = conventional_eval.iloc[:,0]

# Define the number of bootstrap iterations
n_iterations = 1000

# Define a list to store the performance scores
scores = []

# Perform bootstrapping
for i in range(n_iterations):
    # Generate a bootstrap sample
    x_train_boot, y_train_boot = resample(x_train, y_train)
    
    # Train a binary classifier on the bootstrap sample
    lr = LogisticRegression()
    lr.fit(x_train,y_train)
    
    # Evaluate the performance of the classifier on the test set
    y_pred = lr.predict(x_train)
    score = accuracy_score(y_train, y_pred)
    
    # Store the performance score
    scores.append(score)

# Calculate the mean and standard deviation of the performance scores
mean_score = np.mean(scores)
std_score = np.std(scores)

# Print the results
print("train accuracy: {:.4f} +/- {:.4f}".format(mean_score, std_score))




# Perform bootstrapping
for i in range(n_iterations):
    # Generate a bootstrap sample
    x_train_boot, y_train_boot = resample(x_train, y_train)
    
    # Train a binary classifier on the bootstrap sample
    lr = LogisticRegression()
    lr.fit(x_train,y_train)
    
    # Evaluate the performance of the classifier on the test set
    y_pred = lr.predict(x_dev)
    score = accuracy_score(y_dev, y_pred)
    
    # Store the performance score
    scores.append(score)

# Calculate the mean and standard deviation of the performance scores
mean_score = np.mean(scores)
std_score = np.std(scores)

# Print the results
print("Accuracy: {:.4f} +/- {:.4f}".format(mean_score, std_score))





# Perform bootstrapping
for i in range(n_iterations):
    # Generate a bootstrap sample
    x_train_boot, y_train_boot = resample(x_train, y_train)
    
    # Train a binary classifier on the bootstrap sample
    lr = LogisticRegression()
    lr.fit(x_train,y_train)
    
    # Evaluate the performance of the classifier on the test set
    y_pred = lr.predict(x_eval)
    score = accuracy_score(y_eval, y_pred)
    
    # Store the performance score
    scores.append(score)

# Calculate the mean and standard deviation of the performance scores
mean_score = np.mean(scores)
std_score = np.std(scores)

# Print the results
print("Accuracy: {:.4f} +/- {:.4f}".format(mean_score, std_score))



#%% (E) bagging

#use the entire training data set
conventional_train = pd.read_csv("00_train_03.csv",header=None)
conventional_dev = pd.read_csv("00_dev_03.csv",header=None)
conventional_eval = pd.read_csv("00_eval_03.csv",header=None)

#before using pca we whould train the model and use a logistic
#regression to just see how well it performs
x_train = conventional_train.iloc[:,1:3]
y_train = conventional_train.iloc[:,0]
x_dev = conventional_dev.iloc[:,1:3]
y_dev = conventional_dev.iloc[:,0]
x_eval = conventional_eval.iloc[:,1:3]
y_eval = conventional_eval.iloc[:,0]


#for bag we need random 75% of dataset
conventional_train_shuffle = conventional_train.sample(frac=1)
conv_train_75 = conventional_train_shuffle.iloc[0:75000,:]
x_75_train = conv_train_75.iloc[:,1:3]
y_75_train = conv_train_75.iloc[:,0]

# define the model
model = BaggingClassifier()
model.fit(x_75_train,y_75_train)
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


n_scores = cross_val_score(model, 
                           x_75_train, 
                           y_75_train, 
                           scoring='accuracy', 
                           cv=cv, 
                           n_jobs=-1, 
                           error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


n_scores = cross_val_score(model, 
                           x_dev, 
                           y_dev, 
                           scoring='accuracy', 
                           cv=cv, 
                           n_jobs=-1, 
                           error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


n_scores = cross_val_score(model, 
                           x_eval, 
                           y_eval, 
                           scoring='accuracy', 
                           cv=cv, 
                           n_jobs=-1, 
                           error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

y_pred_proba = model.predict_proba(x_75_train)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_75_train, y_pred_proba)
plt.plot(fpr,tpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")


#%% (F) boosting

#use the entire training data set
conventional_train = pd.read_csv("00_train_03.csv",header=None)
conventional_dev = pd.read_csv("00_dev_03.csv",header=None)
conventional_eval = pd.read_csv("00_eval_03.csv",header=None)

#before using pca we whould train the model and use a logistic
#regression to just see how well it performs
x_train = conventional_train.iloc[:,1:3]
y_train = conventional_train.iloc[:,0]
x_dev = conventional_dev.iloc[:,1:3]
y_dev = conventional_dev.iloc[:,0]
x_eval = conventional_eval.iloc[:,1:3]
y_eval = conventional_eval.iloc[:,0]


pca = PCA()
x_train = pca.fit_transform(x_train)
x_eval = pca.fit_transform(x_eval)
x_dev = pca.fit_transform(x_dev)


lr = LogisticRegression()
lr.fit(x_train,y_train)

ada_boost = AdaBoostClassifier(
    base_estimator=lr, n_estimators = 50, random_state=42)

ada_boost.fit(x_train,y_train)

ada_model = ada_boost.fit(x_train,y_train)

y_pred_trainf = ada_boost.predict(x_train)
y_pred_evalf = ada_boost.predict(x_eval)
y_pred_devf = ada_boost.predict(x_dev)

train_error = 1- (metrics.accuracy_score(y_train,y_pred_trainf))
dev_error = 1- (metrics.accuracy_score(y_dev,y_pred_devf))
eval_error = 1- (metrics.accuracy_score(y_eval,y_pred_evalf))

print("train error score is: ",train_error)
print("dev error score is: ",dev_error)
print("eval error score is: ",eval_error)

y_pred_proba = ada_boost.predict_proba(x_75_train)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_75_train, y_pred_proba)
plt.plot(fpr,tpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")


#%%
import matplotlib.patches as mpatches

y_pred_proba = model.predict_proba(x_75_train)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_75_train, y_pred_proba)
plt.plot(fpr,tpr)

y_pred_proba = ada_boost.predict_proba(x_75_train)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_75_train, y_pred_proba)
plt.plot(fpr,tpr)

y_pred_proba = lr.predict_proba(x_train)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_train, y_pred_proba)
plt.plot(fpr,tpr)
                               
green_patch = mpatches.Patch(color='green', label='bootstrap')
blue_patch = mpatches.Patch(color='blue', label='conventional pca')
orange_patch = mpatches.Patch(color='orange', label='ada_boost')
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(handles=[blue_patch,orange_patch,green_patch])














