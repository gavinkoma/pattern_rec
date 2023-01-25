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
from sklearn.metrics import confusion_matrix, classification_report, precision_score
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


qda = QuadraticDiscriminantAnalysis()
model = qda.fit(X_train,y_train)
print(model.priors_)
print(model.means_)
pred_train = model.predict(X_train)
print(metrics.accuracy_score(y_train,pred_train))

#so now we should look at the prediction
pred = model.predict(X_test)
print(np.unique(pred,return_counts=True))
print(confusion_matrix(pred, y_test))
print(classification_report(y_test,pred,digits=4))

#so we get an accuracy of 0.792

#%%ok so we should do a custom one
#we primarily focused on simple linear ones in class, so lets do that one to start
#we can visualize the data as well to get an idea
#errr actually lets just do a gaussian nb analysis

df_train = pd.read_csv('train.csv',
                       header=4,
                       names=["animal","xvec","yvec"])

print(df_train.info()) #same data as before, no unnamed columns either which is good

dog = df_train.loc[df_train['animal'] == 'dogs']
cat = df_train.loc[df_train['animal'] == 'cats']

print(len(df_train)) #double check length
print(len(dog)+len(cat))

#we can plot now! 
#visualization is important!

plt.title("Dog vs. Cat")
plt.xlabel("xvec")
plt.ylabel("yvec")
plt.scatter(dog.xvec,dog.yvec,color='blue',alpha=0.3)
plt.scatter(cat.xvec,cat.yvec,color='red',alpha=0.3)#looks good so far

#so we need to classify dogs and cats in a binary manner 
#uhhh so dogs will be 1 because theyre better than cats

df_train.animal=[1 if i =="dogs" else 0 for i in df_train.animal]

x = df_train.drop(['animal'],axis = 1)
y = df_train.animal.values

#and then i guess im just going to use sklearn for gaussian analysis?
#i dont think the homework wants me to write my own code entirely
#although unsure

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

print("NB Score: ",nb.score(x_train,y_train))

print("NB Score: ",nb.score(x_test,y_test))


#%%compute and plot error rate

#make a list of samples of prior in range [0,1] in steps of 0.01
#cat is 1-P("dog")
prior_dog = np.arange(0.0,1.01,0.01)

#make list for 1-p(dog)
p_cat = []
for val in np.nditer(prior_dog):
    p_cat.append(1-val)
    

#so we will want to calculate the posterior
def compute_posterior(prior,sensitivity,specificity):
    likelihood = sensitivity
    marginal_likelihood = sensitivity*prior+(1-specificity)*(1-prior)
    posterior = (likelihood*prior)/marginal_likelihood
    return(posterior)

posterior_values = compute_posterior(prior_dog,sensitivity=0.99,specificity=0.99)
plt.plot(prior_dog,posterior_values)
    
    
    
    
    



