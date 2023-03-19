#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:13:12 2023

@author: gavinkoma
"""

import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework8/data/data10')
print(os.getcwd())

train = pd.read_csv("train.csv",header=None)
dev = pd.read_csv("dev.csv",header=None)
evalu = pd.read_csv("eval.csv",header=None)

#%%hw pt1
#we need to test on /dev, /train, and /eval
labels_dev = dev.iloc[:,0] #label column
labels_train = train.iloc[:,0] #label column
labels_eval = evalu.iloc[:,0] #label column

xvalues_dev = pd.DataFrame([dev.iloc[:,1],dev.iloc[:,2]]).T
xvalues_train = pd.DataFrame([train.iloc[:,1],train.iloc[:,2]]).T
xvalues_eval = pd.DataFrame([evalu.iloc[:,1],evalu.iloc[:,2]]).T

training_data = xvalues_dev
label_data = labels_dev

neighbors_val = np.arange(1,10)

train_accuracy = np.empty(len(neighbors_val))

dev_test_accuracy = np.empty(len(neighbors_val))
train_test_accuracy = np.empty(len(neighbors_val))
eval_test_accuracy = np.empty(len(neighbors_val))


for i, k in enumerate(neighbors_val):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(training_data,label_data)
    
    #compute data using /dev + /train for /train /dev /eval
    train_accuracy[i] = knn.score(training_data,label_data)
    
    dev_test_accuracy[i] = knn.score(xvalues_dev, labels_dev)    
    train_test_accuracy[i] = knn.score(xvalues_train, labels_train)    
    eval_test_accuracy[i] = knn.score(xvalues_eval, labels_eval)

    
#generate plot
plt.figure()

plt.plot(neighbors_val, train_accuracy, label = 'Training Compiled dataset Accuracy')

plt.plot(neighbors_val, dev_test_accuracy, label = 'dev dataset accuracy')
plt.plot(neighbors_val, train_test_accuracy, label = 'train dataset accuracy')
plt.plot(neighbors_val, eval_test_accuracy, label = 'eval dataset accuracy')

print(train_accuracy)

plt.legend()
plt.title("dataset10")
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

#%% hw1 pt2
#use kmn now but use the same amount of clusters per class, only use /dev 
#for this training session to find best performance and use it to
#classify /dev /train and /eval

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300,n_init=10,random_state=0)
    kmeans.fit(training_data)
    wcss.append(kmeans.inertia_)
    
plt.figure()
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


plt.figure()
kmeans = KMeans(n_clusters=4, init='k-means++', 
                max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(training_data)
plt.scatter(training_data.iloc[:,0], training_data.iloc[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('KMeans Graph with 4 Clusters')
plt.xlabel("feature 1 vector")
plt.ylabel("feature 2 vector")
plt.show()

#%%

scoredev = metrics.accuracy_score(labels_dev,kmeans.predict(xvalues_dev))
scoretrain = metrics.accuracy_score(labels_train,kmeans.predict(xvalues_train))
scoreeval = metrics.accuracy_score(labels_eval,kmeans.predict(xvalues_eval))

print("dev accuracy score: " + str(scoredev) +'\n\n')
print("train accuracy score: " + str(scoretrain) +'\n\n')
print("eval accuracy score: " + str(scoreeval) +'\n\n')




