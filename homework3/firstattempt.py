#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:07:50 2023

@author: gavinkoma
"""
#homework 3
#due on 02/06

#%%import modules
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

#%%
#generate 11 independent sets of random data consisting of 10^6 points
#from a 1D GRV with a variance of 1. make sure to use a mean of 1 +- delta
#where delta will vary from 0.9 to 1.0 in equal steps of 0.2

#start with generation 
mu = 1 
var = 1**2
deltrange = np.arange(0.9,1.1,0.02)
output = [round(float(x),2) for x in deltrange]
        
d = {}
for val in output:
    d["dataframe{0}".format(val)]=np.random.normal(loc=val,scale=var,size=1000000)

    

#%%
#for set with mean 1 --> estimate with max likelihood
#plot and use log scale
#compute mean and repeat by taking all average of N for first 6
#is the second estimate biased? 
#pull data with mean 1
for i, data in d.items():
    mean = sum(data)/len(data)
    print("Mean of the dataset with delta value {val} is: {kval}".format(val=i,kval=mean))

print("Mean of data set with delta of 1.00 is: 0.99736")


#%% 
#we should plot 
values = d['dataframe1.0']
count = [(count+1,value) for count,value in enumerate(values)]
df1 = pd.DataFrame(count,columns=['N','mean'])
#print(df1.head())
df1['mean'].mean()
print(df1['mean'].mean())

#im not really sure what this plot is supposed to look like...
plot = df1.plot.scatter(x='N', y='mean')

#we need to define the samples
samples = [1,5,10**1,50,10**2,500,10**3,5000,10**4,50000,10**5,500000,10**6]
new_dict = {}
for i in samples:
    data = [random.gauss(1,var) for j in range (i)]
    new_dict[i] = data

#delta mean
delta_mean = []
for i,data in new_dict.items():
    mean = sum(data)/len(data)
    delta_mean.append(mean)

plot = list(zip(samples,delta_mean))
df2 = pd.DataFrame(plot, columns=['N_val','meanestimate'])
print(df2.head())

fig,ax = plt.subplots(figsize=(9,6))
ax.scatter(df2['N_val'],df2['meanestimate'],s = 80,color='r')
ax.set_xscale("Log")


#%%
#we need the means for the first 6 datasets
#weve already made all 11 data sets
#but we do need to make a dict that has the mean values in it

dict_val = {} #dixie chicks started on my playlist, motivation is back

for i in deltrange:
    dataset = {}
    for j in samples:
        key = "{}".format(j)
        value = [random.gauss(i,var) for k in range(j)]
        dataset[key]=value
    dict_val["mean_{}".format(i)] = dataset

print(dict_val['mean_0.9']['10'])

#%%
#so now we have to do it fall of the value 
#which would be 11x13 which is 143

means = []
for name,data in dict_val.items():
    for key, val in dataset.items():
        meanval = np.mean(values)
        means.append((name,key,mean))

df_mean = pd.DataFrame(means,columns=['means','n','mean_estimate'])
print(df_mean.tail(10))
df_mean.dtypes
df_mean["n"]=df_mean["n"].astype("float")







