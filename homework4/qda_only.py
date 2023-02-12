#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:53:39 2023

@author: gavinkoma
"""

#%% start with modules
import glob
import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pylab as pl
from scipy import linalg


#%%
os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework4/data')
print(os.getcwd())

files = glob.glob('*8*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 

#were going to start this with just data 8 because im getting confused
#with the dynamic loops i tried to do earlier

#lets organize the data
df_train = pd.DataFrame(d_all['data8train.csv'])
df_train.columns = ['labels','vec1','vec2']

df_eval = pd.DataFrame(d_all['data8eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

# run qda
qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train,y_train)
print(model.priors_)
print(model.means_)

pred_train = model.predict(x_train)
print("score for the training data:\n", metrics.accuracy_score(y_train, pred_train))
#look at prediction
pred = model.predict(x_test)
print(np.unique(pred,return_counts=True))
print(confusion_matrix(pred,y_test))
print("the score for the test data:",classification_report(y_test, pred, digits=4))
      
# run qda on dev data

df_eval = pd.DataFrame(d_all['data8dev.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train,y_train)
print(model.priors_)
print(model.means_)

#we need to plot
plt.scatter(x_test.iloc[:,0],
            x_test.iloc[:,1],
            c=y_test,
            alpha=0.7)
plt.title('QDA (Test set) for Dataset 8')
plt.xlim(-4,4)
plt.ylim(-4,4)
      
#%%
#do the same for 9
os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework4/data')
print(os.getcwd())

files = glob.glob('*9*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 

#were going to start this with just data 8 because im getting confused
#with the dynamic loops i tried to do earlier
#lets organize the data
df_train = pd.DataFrame(d_all['data9train.csv'])
df_train.columns = ['labels','vec1','vec2']

df_eval = pd.DataFrame(d_all['data9eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

# run qda
qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train,y_train)
print(model.priors_)
print(model.means_)

pred_train = model.predict(x_train)
print("score for the training data:\n", metrics.accuracy_score(y_train, pred_train))
#look at prediction
pred = model.predict(x_test)
print(np.unique(pred,return_counts=True))
print(confusion_matrix(pred,y_test))
print("the score for the test data:\n",classification_report(y_test, pred, digits=4))
      
# run qda on dev data

df_eval = pd.DataFrame(d_all['data9dev.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train,y_train)
print(model.priors_)
print(model.means_)

pred_train = model.predict(x_train)
print("score for the training data:\n", metrics.accuracy_score(y_train, pred_train))
#look at prediction
pred = model.predict(x_test)
print(np.unique(pred,return_counts=True))
print(confusion_matrix(pred,y_test))
print("the score for the dev data:\n",classification_report(y_test, pred, digits=4))


plt.scatter(x_test.iloc[:,0],
            x_test.iloc[:,1],
            c=y_test,
            alpha=0.7)
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title('QDA (Test set) for Dataset 9')
      

#%% for data 10
os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework4/data')
print(os.getcwd())

files = glob.glob('*10*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 

#were going to start this with just data 8 because im getting confused
#with the dynamic loops i tried to do earlier
#lets organize the data
df_train = pd.DataFrame(d_all['data10train.csv'])
df_train.columns = ['labels','vec1','vec2']

df_eval = pd.DataFrame(d_all['data10eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

# run qda
qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train,y_train)
print(model.priors_)
print(model.means_)

pred_train = model.predict(x_train)
print("score for the training data:\n", metrics.accuracy_score(y_train, pred_train))
#look at prediction
pred = model.predict(x_test)
print(np.unique(pred,return_counts=True))
print(confusion_matrix(pred,y_test))
print("the score for the test data:\n",classification_report(y_test, pred, digits=4))
      
# run qda on dev data

df_eval = pd.DataFrame(d_all['data10dev.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train,y_train)
print(model.priors_)
print(model.means_)

pred_train = model.predict(x_train)
print("score for the training data:\n", metrics.accuracy_score(y_train, pred_train))
#look at prediction
pred = model.predict(x_test)
print(np.unique(pred,return_counts=True))
print(confusion_matrix(pred,y_test))
print("the score for the dev data:\n",classification_report(y_test, pred, digits=4))

c = ListedColormap(('red', 'green', 'blue'))

plt.scatter(x_test.iloc[:,0],
            x_test.iloc[:,1],
            c=y_test,
            alpha=0.7)

plt.ylim(-4,4)
plt.xlim(-4,4)
plt.title('QDA (Test set) for Dataset 10')

#%%okay so back to data 8 but concat train & dev as the training data

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework4/data')
print(os.getcwd())

files = glob.glob('*8*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 

#were going to start this with just data 8 because im getting confused
#with the dynamic loops i tried to do earlier
#lets organize the data
df_train = pd.DataFrame(d_all['data8train.csv'])
df_train.columns = ['labels','vec1','vec2']
df_dev = pd.DataFrame(d_all['data8dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
df_dev_train = [df_train,df_dev]
df_dev_train = pd.concat(df_dev_train)

df_eval = pd.DataFrame(d_all['data8eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_dev_train[:][['vec1','vec2']]
y_train = df_dev_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

#im confusing myself with these concats
qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train,y_train)




x_train = pd.DataFrame(d_all['data8train.csv'])
x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']
pred_train_only = model.predict(x_train)
print("score for training data:\n", metrics.accuracy_score(y_train, pred_train_only))



dev_train = pd.DataFrame(d_all['data8dev.csv'])
x_train = df_dev_train[:][['vec1','vec2']]
y_train = df_dev_train[:]['labels']
pred_dev_only = model.predict(x_train)
print("score for training data:\n", metrics.accuracy_score(y_train, pred_dev_only))


pred_test = model.predict(x_test)
print(np.unique(pred_test,return_counts=True))
print(confusion_matrix(pred_test,y_test))
print("score of test data only:\n",classification_report(y_test, pred_test,digits=4))


#%%okay so back to data 9 but concat train & dev as the training data

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework4/data')
print(os.getcwd())

files = glob.glob('*9*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 

#were going to start this with just data 8 because im getting confused
#with the dynamic loops i tried to do earlier
#lets organize the data
df_train = pd.DataFrame(d_all['data9train.csv'])
df_train.columns = ['labels','vec1','vec2']
df_dev = pd.DataFrame(d_all['data9dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
df_dev_train = [df_train,df_dev]
df_dev_train = pd.concat(df_dev_train)

df_eval = pd.DataFrame(d_all['data9eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_dev_train[:][['vec1','vec2']]
y_train = df_dev_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

#im confusing myself with these concats
qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train,y_train)




x_train = pd.DataFrame(d_all['data9train.csv'])
x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']
pred_train_only = model.predict(x_train)
print("score for training data:\n", metrics.accuracy_score(y_train, pred_train_only))



dev_train = pd.DataFrame(d_all['data9dev.csv'])
x_train = df_dev_train[:][['vec1','vec2']]
y_train = df_dev_train[:]['labels']
pred_dev_only = model.predict(x_train)
print("score for training data:\n", metrics.accuracy_score(y_train, pred_dev_only))


pred_test = model.predict(x_test)
print(np.unique(pred_test,return_counts=True))
print(confusion_matrix(pred_test,y_test))
print("score of test data only:\n",classification_report(y_test, pred_test,digits=4))



#%%okay so back to data 10 but concat train & dev as the training data

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework4/data')
print(os.getcwd())

files = glob.glob('*10*.csv') #make a list of all the csv files

d_all = {} #init dict
for file in files: #loop through .csv names
    #make a dict for all of them and add them to use later
    #there are no headers for any of this stuff, dont forget
    d_all["{0}".format(file)] = pd.read_csv(file,header=None) 

#were going to start this with just data 8 because im getting confused
#with the dynamic loops i tried to do earlier
#lets organize the data
df_train = pd.DataFrame(d_all['data10train.csv'])
df_train.columns = ['labels','vec1','vec2']
df_dev = pd.DataFrame(d_all['data10dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
df_dev_train = [df_train,df_dev]
df_dev_train = pd.concat(df_dev_train)

df_eval = pd.DataFrame(d_all['data10eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_dev_train[:][['vec1','vec2']]
y_train = df_dev_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

#im confusing myself with these concats
qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train,y_train)




x_train = pd.DataFrame(d_all['data10train.csv'])
x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']
pred_train_only = model.predict(x_train)
print("score for training data:\n", metrics.accuracy_score(y_train, pred_train_only))



dev_train = pd.DataFrame(d_all['data10dev.csv'])
x_train = df_dev_train[:][['vec1','vec2']]
y_train = df_dev_train[:]['labels']
pred_dev_only = model.predict(x_train)
print("score for training data:\n", metrics.accuracy_score(y_train, pred_dev_only))


pred_test = model.predict(x_test)
print(np.unique(pred_test,return_counts=True))
print(confusion_matrix(pred_test,y_test))
print("score of test data only:\n",classification_report(y_test, pred_test,digits=4))







