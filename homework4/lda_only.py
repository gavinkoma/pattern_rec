#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:43:09 2023

@author: gavinkoma
"""
#%% start with modules
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
#%%
#lda wont be too bad, weve already visualized all the graphs as well 
#lets start with 8
#lets organize the data
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

#itll be important to not forget that LDA only has one line as a boundary
#and its gotta be linear!

model = LinearDiscriminantAnalysis()
model.fit(x_train,y_train)

#use model and make prediction
#we need to eval also

df_dev = pd.DataFrame(d_all['data8dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']

cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)


scores_train = cross_val_score(model, x_train,y_train,scoring='accuracy',
                         cv=cv,n_jobs=1)
y_train_pred = model.predict(x_train)


scores_dev=cross_val_score(model, x_dev,y_dev,scoring='accuracy',
                         cv=cv,n_jobs=1)
y_dev_pred = model.predict(x_dev)


scores_eval=cross_val_score(model, x_test,y_test,scoring='accuracy',
                         cv=cv,n_jobs=1)
y_test_pred = model.predict(x_test)

print("train score: ",np.mean(scores_train))
print("dev score: ",np.mean(scores_dev))
print("test score: ",np.mean(scores_eval))


plt.scatter(x_test.iloc[:,0],
            x_test.iloc[:,1],
            c=y_test,
            alpha=0.7)
plt.title('LDA (Test set) for Dataset 8')
plt.xlim(-4,4)
plt.ylim(-4,4)


#%%
#lda wont be too bad, weve already visualized all the graphs as well 
#lets start with 9
#lets organize the data
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

#itll be important to not forget that LDA only has one line as a boundary
#and its gotta be linear!

model = LinearDiscriminantAnalysis()
model.fit(x_train,y_train)

#use model and make prediction
#we need to eval also

df_dev = pd.DataFrame(d_all['data9dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']

cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)


scores_train = cross_val_score(model, x_train,y_train,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_dev=cross_val_score(model, x_dev,y_dev,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_eval=cross_val_score(model, x_test,y_test,scoring='accuracy',
                         cv=cv,n_jobs=1)

print("train score: ",np.mean(scores_train))
print("dev score: ",np.mean(scores_dev))
print("test score: ",np.mean(scores_eval))

plt.scatter(x_test.iloc[:,0],
            x_test.iloc[:,1],
            c=y_test,
            alpha=0.7)
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title('LDA (Test set) for Dataset 9')

#%%
#lda wont be too bad, weve already visualized all the graphs as well 
#lets start with 10
#lets organize the data
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

#itll be important to not forget that LDA only has one line as a boundary
#and its gotta be linear!

model = LinearDiscriminantAnalysis()
model.fit(x_train,y_train)

#use model and make prediction
#we need to eval also

df_dev = pd.DataFrame(d_all['data10dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']

cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)


scores_train = cross_val_score(model, x_train,y_train,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_dev=cross_val_score(model, x_dev,y_dev,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_eval=cross_val_score(model, x_test,y_test,scoring='accuracy',
                         cv=cv,n_jobs=1)

print("train score: ",np.mean(scores_train))
print("dev score: ",np.mean(scores_dev))
print("test score: ",np.mean(scores_eval))

plt.scatter(x_test.iloc[:,0],
            x_test.iloc[:,1],
            c=y_test,
            alpha=0.7)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.title('LDA (Test set) for Dataset 10')


#%% okay so now we /need/ to train on both dev & train
#lda wont be too bad, weve already visualized all the graphs as well 
#lets start with 8
#lets organize the data
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
x_dev_train = df_dev_train[['vec1','vec2']]
y_dev_train = df_dev_train['labels']


df_train = pd.DataFrame(d_all['data8train.csv'])
df_train.columns = ['labels','vec1','vec2']

df_eval = pd.DataFrame(d_all['data8eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

#itll be important to not forget that LDA only has one line as a boundary
#and its gotta be linear!

model = LinearDiscriminantAnalysis()
model.fit(x_dev_train,y_dev_train)

#use model and make prediction
#we need to eval also

df_dev = pd.DataFrame(d_all['data8dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']

cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)


scores_train = cross_val_score(model, x_train,y_train,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_dev=cross_val_score(model, x_dev,y_dev,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_eval=cross_val_score(model, x_test,y_test,scoring='accuracy',
                         cv=cv,n_jobs=1)

print("train score: ",np.mean(scores_train))
print("dev score: ",np.mean(scores_dev))
print("test score: ",np.mean(scores_eval))


#%% okay so now we /need/ to train on both dev & train
#lda wont be too bad, weve already visualized all the graphs as well 
#lets start with 9
#lets organize the data
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
x_dev_train = df_dev_train[['vec1','vec2']]
y_dev_train = df_dev_train['labels']


df_train = pd.DataFrame(d_all['data9train.csv'])
df_train.columns = ['labels','vec1','vec2']

df_eval = pd.DataFrame(d_all['data9eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

#itll be important to not forget that LDA only has one line as a boundary
#and its gotta be linear!

model = LinearDiscriminantAnalysis()
model.fit(x_dev_train,y_dev_train)

#use model and make prediction
#we need to eval also

df_dev = pd.DataFrame(d_all['data9dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']

cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)


scores_train = cross_val_score(model, x_train,y_train,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_dev=cross_val_score(model, x_dev,y_dev,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_eval=cross_val_score(model, x_test,y_test,scoring='accuracy',
                         cv=cv,n_jobs=1)

print("train score: ",np.mean(scores_train))
print("dev score: ",np.mean(scores_dev))
print("test score: ",np.mean(scores_eval))


#%% okay so now we /need/ to train on both dev & train
#lda wont be too bad, weve already visualized all the graphs as well 
#lets start with 9
#lets organize the data
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
x_dev_train = df_dev_train[['vec1','vec2']]
y_dev_train = df_dev_train['labels']


df_train = pd.DataFrame(d_all['data10train.csv'])
df_train.columns = ['labels','vec1','vec2']

df_eval = pd.DataFrame(d_all['data10eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

#itll be important to not forget that LDA only has one line as a boundary
#and its gotta be linear!

model = LinearDiscriminantAnalysis()
model.fit(x_dev_train,y_dev_train)

#use model and make prediction
#we need to eval also

df_dev = pd.DataFrame(d_all['data10dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']

cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)


scores_train = cross_val_score(model, x_train,y_train,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_dev=cross_val_score(model, x_dev,y_dev,scoring='accuracy',
                         cv=cv,n_jobs=1)

scores_eval=cross_val_score(model, x_test,y_test,scoring='accuracy',
                         cv=cv,n_jobs=1)

print("train score: ",np.mean(scores_train))
print("dev score: ",np.mean(scores_dev))
print("test score: ",np.mean(scores_eval))


