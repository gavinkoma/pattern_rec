#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:47:10 2023

@author: gavinkoma
"""

#%% start with modules
import pandas as pd
import glob
import os
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
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

df_dev = pd.DataFrame(d_all['data8dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']


sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

pca = PCA(n_components=2)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred_train = classifier.predict(x_train)
print("train accuracy:\n",metrics.accuracy_score(y_train,y_pred_train))

y_pred_dev = classifier.predict(x_dev)
print("dev accuracy:\n",metrics.accuracy_score(y_dev,y_pred_dev))

y_pred_test = classifier.predict(x_test)
print("test accuracy:\n",metrics.accuracy_score(y_test,y_pred_test))

#we should plot the decision bound

x_set, y_set = x_test,y_test

x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                     stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1,
                     stop = x_set[:, 1].max() + 1, step = 0.01))
  
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),
             x2.ravel()]).T).reshape(x1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
  
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
  
# title for scatter plot
plt.title('CI-PCA (Test set) for Dataset 8') 
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.legend()



#%% data 9
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

df_dev = pd.DataFrame(d_all['data9dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']


sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

pca = PCA(n_components=2)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred_train = classifier.predict(x_train)
print("train accuracy:\n",metrics.accuracy_score(y_train,y_pred_train))

y_pred_dev = classifier.predict(x_dev)
print("dev accuracy:\n",metrics.accuracy_score(y_dev,y_pred_dev))

y_pred_test = classifier.predict(x_test)
print("test accuracy:\n",metrics.accuracy_score(y_test,y_pred_test))

#we should plot the decision bound

x_set, y_set = x_test,y_test

x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                     stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1,
                     stop = x_set[:, 1].max() + 1, step = 0.01))
  
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),
             x2.ravel()]).T).reshape(x1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
  
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
  
# title for scatter plot
plt.title('CI-PCA (Test set) for Dataset 9') 
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.legend()
  

#%% data 10
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

df_eval = pd.DataFrame(d_all['data10eval.csv'])
df_eval.columns = ['labels','vec1','vec2']

x_train = df_train[:][['vec1','vec2']]
y_train = df_train[:]['labels']

x_test = df_eval[:][['vec1','vec2']]
y_test = df_eval[:]['labels']

df_dev = pd.DataFrame(d_all['data10dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']


sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

pca = PCA(n_components=2)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred_train = classifier.predict(x_train)
print("train accuracy:\n",metrics.accuracy_score(y_train,y_pred_train))

y_pred_dev = classifier.predict(x_dev)
print("dev accuracy:\n",metrics.accuracy_score(y_dev,y_pred_dev))

y_pred_test = classifier.predict(x_test)
print("test accuracy:\n",metrics.accuracy_score(y_test,y_pred_test))

#we should plot the decision bound

x_set, y_set = x_test,y_test

x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                     stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1,
                     stop = x_set[:, 1].max() + 1, step = 0.01))
  
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),
             x2.ravel()]).T).reshape(x1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
  
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
  
# title for scatter plot
plt.title('CI-PCA (Test set) for Dataset 10') 
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.ylim(-4,4)
plt.xlim(-4,4)
plt.legend()
  

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

df_dev = pd.DataFrame(d_all['data8dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']

### everything is labeled
sc=StandardScaler()
x_train = sc.fit_transform(x_dev_train)
x_test = sc.transform(x_test)

pca = PCA(n_components=2)

x_train = pca.fit_transform(x_dev_train)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_

classifier = LogisticRegression(random_state=0)
classifier.fit(x_dev_train,y_dev_train)

y_pred_train = classifier.predict(x_dev_train)
print("train accuracy:\n",metrics.accuracy_score(y_dev_train,y_pred_train))

y_pred_dev = classifier.predict(x_dev)
print("dev accuracy:\n",metrics.accuracy_score(y_dev,y_pred_dev))

y_pred_test = classifier.predict(x_test)
print("test accuracy:\n",metrics.accuracy_score(y_test,y_pred_test))

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

df_dev = pd.DataFrame(d_all['data9dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']

### everything is labeled
sc=StandardScaler()
x_train = sc.fit_transform(x_dev_train)
x_test = sc.transform(x_test)

pca = PCA(n_components=2)

x_train = pca.fit_transform(x_dev_train)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_

classifier = LogisticRegression(random_state=0)
classifier.fit(x_dev_train,y_dev_train)

y_pred_train = classifier.predict(x_dev_train)
print("train accuracy:\n",metrics.accuracy_score(y_dev_train,y_pred_train))

y_pred_dev = classifier.predict(x_dev)
print("dev accuracy:\n",metrics.accuracy_score(y_dev,y_pred_dev))

y_pred_test = classifier.predict(x_test)
print("test accuracy:\n",metrics.accuracy_score(y_test,y_pred_test))

#%%
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

df_dev = pd.DataFrame(d_all['data10dev.csv'])
df_dev.columns = ['labels','vec1','vec2']
x_dev = df_dev[:][['vec1','vec2']]
y_dev = df_dev[:]['labels']

### everything is labeled
sc=StandardScaler()
x_train = sc.fit_transform(x_dev_train)
x_test = sc.transform(x_test)

pca = PCA(n_components=2)

x_train = pca.fit_transform(x_dev_train)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_

classifier = LogisticRegression(random_state=0)
classifier.fit(x_dev_train,y_dev_train)

y_pred_train = classifier.predict(x_dev_train)
print("train accuracy:\n",metrics.accuracy_score(y_dev_train,y_pred_train))

y_pred_dev = classifier.predict(x_dev)
print("dev accuracy:\n",metrics.accuracy_score(y_dev,y_pred_dev))

y_pred_test = classifier.predict(x_test)
print("test accuracy:\n",metrics.accuracy_score(y_test,y_pred_test))