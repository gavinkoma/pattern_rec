#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:50:41 2023

@author: gavinkoma
"""

#%%import modules
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

#%%
#okay so we need to do qda in sklearn
#and some custom implementation of a gaussian classfiier
#we need to also assume that the priors are not equal, so
#in this case we should write a loop that samples the priors 

#import eval and train data points
df_train = pd.read_csv('train.csv',
                       header=4,
                       names=["animal","xvec","yvec"])

df_eval = pd.read_csv('eval.csv',
                      header=4,
                      names=["animal","xvec","yvec"])

X_train = df_train[:][['xvec','yvec']]
y_train = df_train[:]['animal']

X_test = df_eval[:][['xvec','yvec']]
y_test = df_eval[:]['animal']


qda = QuadraticDiscriminantAnalysis(priors=[0.5,0.5])
model = qda.fit(X_train,y_train)
print(model.priors_)
print(model.means_)
pred_train = model.predict(X_train)
print("the score for training data:\n", metrics.accuracy_score(y_train,pred_train))

#so now we should look at the prediction
pred = model.predict(X_test)
print(np.unique(pred,return_counts=True))
print(confusion_matrix(pred, y_test))
print("the score for test data:",classification_report(y_test,pred,digits=4))

#so we get an accuracy of 0.792


#%%ok so we should do a custom one
#we primarily focused on simple linear ones in class, so lets do that one to start
#we can visualize the data as well to get an idea
#errr actually lets just do a gaussian nb analysis

df_train = pd.read_csv('train.csv',
                       header=4,
                       names=["animal","xvec","yvec"])
df_eval = pd.read_csv('eval.csv',
                      header=4,
                      names=["animal","xvec","yvec"])

dog_train = df_train.loc[df_train['animal'] == 'dogs']
cat_train = df_train.loc[df_train['animal'] == 'cats']
dog_eval = df_eval.loc[df_eval['animal'] == 'dogs']
cat_eval = df_eval.loc[df_eval['animal'] == 'cats']


#we can plot now! 
#visualization is important!
plt.figure()
plt.title("Dog vs. Cat Training")
plt.xlabel("xvec")
plt.ylabel("yvec")
plt.scatter(dog_train.xvec,dog_train.yvec,color='blue',alpha=0.3)
plt.scatter(cat_train.xvec,cat_train.yvec,color='red',alpha=0.3)#looks good so far

plt.figure()
plt.title("Dog vs. Cat Eval")
plt.ylabel("yvec")
plt.scatter(dog_eval.xvec,dog_eval.yvec,color='blue',alpha=0.3)
plt.scatter(cat_eval.xvec,cat_eval.yvec,color='red',alpha=0.3)#looks good so far


#so we need to classify dogs and cats in a binary manner 
#uhhh so dogs will be 1 because theyre better than cats
df_train.animal=[1 if i =="dogs" else 0 for i in df_train.animal]
df_eval.animal=[1 if i =="dogs" else 0 for i in df_eval.animal]
x_train = df_train.drop(['animal'],axis = 1)
y_train = df_train.animal.values
x_test = df_eval.drop(['animal'],axis = 1)
y_test = df_eval.animal.values


#and then i guess im just going to use sklearn for gaussian analysis?
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nb.fit(x_test,y_test)

print("NB Score: ",nb.score(x_train,y_train))
print("NB Score: ",nb.score(x_test,y_test))


#%%compute and plot error rate
#import eval and train data points
df_train = pd.read_csv('train.csv',
                       header=4,
                       names=["animal","xvec","yvec"])

df_eval = pd.read_csv('eval.csv',
                      header=4,
                      names=["animal","xvec","yvec"])

X_train = df_train[:][['xvec','yvec']]
y_train = df_train[:]['animal']

X_test = df_eval[:][['xvec','yvec']]
y_test = df_eval[:]['animal']


#make a list of samples of prior in range [0,1] in steps of 0.01
#cat is 1-P("dog")
prior_dog = np.arange(0.0,1.01,0.01)

p_cat = []#make list for 1-p(dog)
for val in np.nditer(prior_dog):
    p_cat.append(1-val)
    
p_cat = np.array(p_cat)
priorval = np.array((p_cat,prior_dog)).T

#when we calculate priors we need to assess
#not entirely sure what the model he wants us to use but im assuming he wants qda?

#lets just start by repeating our previous code?
df_train = df_train
df_eval = df_eval
x_train = X_train
print(len(x_train))
y_train = y_train
print(len(y_train))
x_test = X_test
y_test = y_test

#we need to forloop through the qda analysis and do it for every
# #value of the priors but idk why this isnt working
error_values = []

for val,cval in priorval:
        qda_loop = QuadraticDiscriminantAnalysis(priors=[val,cval])
        print(val,cval)
        model_loop = qda_loop.fit(x_train,y_train)
        #we only want to do this for the evaluation data (x_test)
        pred_loop = model_loop.predict(x_test)
        error_loop = metrics.accuracy_score(y_test,pred_loop)
        error_values.append(error_loop)

plt.title("Performance as a Function of Priors")
plt.xlabel("Prior Probability")
plt.ylabel("Accuracy//Error")
plt.scatter(prior_dog,error_values,color='blue',alpha=0.3)

         


