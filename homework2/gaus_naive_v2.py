#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:41:26 2023

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


#%%predictions
vectors = x_train #these are our samples
labels = label_train #these are the binary encoded labels

priors = {}
means = {}
covs = {}
classes = np.unique(labels) #our two unique classes, could vary if had more

for unique in classes:
    vectors_unique = vectors[labels == unique] #store a values according the size/length
    priors = {0:0.5,1:0.5} #assume priors are equal
    means[unique] = np.mean(vectors_unique,axis=0) #calculate the means of each vector
    #need to calculate the covariance of the vectors that belong to the class
    covs[unique] = np.cov(vectors_unique,rowvar=False) #each row /= variable, rowvar=false
    
    
#okay define the prediction function
def prediction(vectors):
    predictions=[] #store predictions here
    for vec in vectors:
        posteriors = [] #store posteriors here please
        for unique in classes:
            #priors will be the same for this
            prior = np.log(priors[unique])#calculate log of the unique priors per class
            #calculate the inverse cov matrix, 1 per unique class
            inv_cov = np.linalg.inv(covs[unique])
            #calcualte the determinant of the inv matrix
            inv_cov_det = np.linalg.det(inv_cov)
            #we will have to calculate the difference between
            #each value and the mean --> difference!
            difference = vec-means[unique]
            #1/2(invcovdet)*1/2(transposed differences)*inverse cov * differences
            #the @ symbol is just matrix multiplication
            likelihood = 0.5*np.log(inv_cov_det) - 0.5*difference.T @ inv_cov @ difference
            posterior = prior + likelihood
            posteriors.append(posterior)
        
        pred = classes[np.argmax(posteriors)]
        predictions.append(pred)
    #return predictions for later use --> numpy array
    return np.array(predictions)




#%%unchanged
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


