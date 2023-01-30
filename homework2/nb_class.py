#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:55:07 2023

@author: gavinkoma
"""
#okay so we know the skeleton of our naive bayes work
#1. we first have to get all of the summary statistics 
#2. get the probability of the class for each sample
#assume gaussian and independent! (we have two features, 2 class)

#import the modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('seaborn')

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

x_train = df_train[:][['xvec','yvec']]
dataset_train = np.array(x_train)
label_train = np.array(df_train[:]['animal'])

x_test = df_eval[:][['xvec','yvec']]
dataset_test = np.array(x_test)
label_test = np.array(df_eval[:]['animal'])

prior_dog = np.arange(0.0,1.01,0.01)

prior_cat = []#make list for 1-p(dog)
for val in np.nditer(prior_dog):
    prior_cat.append(1-val)
    
prior_cat = np.array(prior_cat)
priorval = np.array((prior_cat,prior_dog)).T

#separate by class first 
def separate_by_class(X,y):
    class_dict = {}
    for i, label in enumerate(y):
        if label in class_dict:
            class_dict[label].append(i)
        else:
            class_dict[label] = [i]
    for k,v in class_dict.items():
        class_dict[k] = X[v]
    return class_dict

def summarizer(data):
    summary = [[np.mean(column),np.std(column)] for column in zip(*data)]
    return summary

#maximum a posterior estimation is next step
def summarize_by_class(class_dict):
    summary_dict = {}
    for k,v in class_dict.items():
        summary_dict[k] = summarizer(v)
    return summary_dict

#okay we have to do do the bayesian work now
#what exactly needs done?
#1. define priors
#2. define the likelihood function as well by using parameters
#3. make our prediction next
#4. and then we can finish the model by using test data

def prior(class_dict,y):
    # prior_dict = {}
    # total_num = len(y)
    # for k,v in class_dict.items():
    #     prior_dict[k] = len(v)/total_num 
    prior_dict = {1:0.5,0:0.5}
    return prior_dict

def likelihood(class_dict,test_instance):
    likelihood_dict = {}
    feature_summary = summarize_by_class(class_dict)
    for k in feature_summary.keys():
        value = feature_summary[k]
        for i, feature in enumerate(value):
            if k in likelihood_dict:
                likelihood_dict[k] *= norm(feature[0], feature[1]).pdf(test_instance[i])
            else:
                likelihood_dict[k] = norm(feature[0], feature[1]).pdf(test_instance[i])
    return likelihood_dict

def make_prediction(training_set,label,testing_instance):
    class_dict = separate_by_class(training_set, label)
    class_probability = prior(class_dict,label)
    likelihood_dict = likelihood(class_dict,testing_instance)
    prediction = {k: class_probability[k] * likelihood_dict[k] for k in class_probability}
    return max(prediction.keys(), key=lambda k: prediction[k])

def naive_bayes(training_set,label,testing_set):
    prediction = []
    for instance in testing_set:
        prediction.append(make_prediction(training_set, label, instance))
    return np.array(prediction)

y_pred = naive_bayes(dataset_train, label_train, dataset_test)
accuracy_score_test = sum(label_test == y_pred)/len(label_test)
print("error rate of evaluation data: ",accuracy_score_test)

y_train_predict = naive_bayes(dataset_train, label_train, dataset_train)
accuracy_score_train = sum(label_train == y_train_predict)/len(label_train)
print("error rate of training data: ",accuracy_score_train)


#%%varying priors
#okay so we need the code to run with 100 varying priors 
#im literally not sure if my computer can handle this computationally because
#it already takes forever to run it just once but i guess we can give it a toss



error_values=[]

for hera, aries in priorval:
        y_pred = naive_bayes(dataset_train, label_train, dataset_test)
        accuracy_score_test = sum(label_test == y_pred)/len(label_test)
        accuracy_score_test = 1-accuracy_score_test
        error_values.append(accuracy_score_test)







