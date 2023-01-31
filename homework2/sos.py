#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:28:30 2023

@author: gavinkoma
"""
#%%import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% data load
#as always, load the data
#read in the data
#lets plot for visualization as well 
df_train = pd.read_csv('train.csv',
                       header=4,
                       names=["animal","xvec","yvec"])
df_eval = pd.read_csv('eval.csv',
                      header=4,
                      names=["animal","xvec","yvec"])

dog_train_plot = df_train.loc[df_train['animal'] == 'dogs']
cat_train_plot = df_train.loc[df_train['animal'] == 'cats']
dog_eval_plot = df_eval.loc[df_eval['animal'] == 'dogs']
cat_eval_plot = df_eval.loc[df_eval['animal'] == 'cats']

plt.figure()
plt.title("Dog vs. Cat Training")
plt.ylabel("yvec")
plt.xlabel("xvec")
plt.scatter(dog_train_plot.xvec,dog_train_plot.yvec,color='blue',alpha=0.3)
plt.scatter(cat_train_plot.xvec,cat_train_plot.yvec,color='red',alpha=0.3)

plt.figure()
plt.title("Dog vs. Cat Testing")
plt.ylabel("yvec")
plt.xlabel("xvec")
plt.scatter(dog_eval_plot.xvec,dog_eval_plot.yvec,color='blue',alpha=0.3)
plt.scatter(cat_eval_plot.xvec,cat_eval_plot.yvec,color='red',alpha=0.3)


#%%
df_train.animal=[1 if i =="dogs" else 0 for i in df_train.animal]
df_eval.animal=[1 if i =="dogs" else 0 for i in df_eval.animal]

#make it all as numpy for consistency
x_train = df_train[:][['xvec','yvec']]
dataset_train = np.array(x_train)
label_train = np.array(df_train[:]['animal'])

x_test = df_eval[:][['xvec','yvec']]
dataset_test = np.array(x_test)
label_test = np.array(df_eval[:]['animal'])

#%%run the prediction function
def predict(samples):
    predictions = []#store predictions
    for val in samples:
        posteriors = []#store posteriors
        for vec in classes:
            #priors will be the same here
            prior = np.log(priors[vec])#calc log of unique priors per class
            #calculate the inverse cov matrix, 1 per unique class
            inv_cov = np.linalg.inv(covs[vec])
            #calculate the determinant of the inv matrix
            inv_cov_det = np.linalg.det(inv_cov)
            #calculate the difference between val & mean
            diff = val-means[vec]
            #1/2(invcovdet)*1/2(transposed differences)*inverse cov * differences
            #the @ symbol is just matrix multiplication           
            likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
            post = prior + likelihood
            posteriors.append(post)
        pred = classes[np.argmax(posteriors)]
        predictions.append(pred)
    #return predictions for accuracy calc
    return np.array(predictions)

# #training
# samples = dataset_train[:,0:2]
# labels = label_train[:]

# priors = {}
# means = {}
# covs = {}
# classes = np.unique(labels) #here we have two unique classes

# for vec in classes:
#     uni_c = samples[labels==vec]#store values
#     priors = {0:0.5,1:0.5}#priors are equal here
#     means[vec] = np.mean(uni_c,axis = 0)#calculate the means of each vector
#     #need to calculate the covariance of the vectors that belong to the class
#     covs[vec] = np.cov(uni_c,rowvar=False)#each row /= variable, rowvar=false
    
# predictions = predict(samples)
# accuracy_score_train = sum(labels == predictions)/len(labels)
# print(accuracy_score_train)

#test
samples = dataset_test[:,0:2]
labels = label_test[:]

priors = {}
means = {}
covs = {}
classes = np.unique(labels) #here we have two unique classes

for vec in classes:
    uni_c = samples[labels==vec]#store values
    priors = {0:0.5,1:0.5}#priors are equal here
    means[vec] = np.mean(uni_c,axis = 0)#calculate the means of each vector
    #need to calculate the covariance of the vectors that belong to the class
    covs[vec] = np.cov(uni_c,rowvar=False)#each row /= variable, rowvar=false
    
predictions = predict(samples)
accuracy_score_test = sum(labels == predictions)/len(labels)
print(accuracy_score_test)



















