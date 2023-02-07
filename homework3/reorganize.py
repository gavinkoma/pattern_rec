#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:59:40 2023

@author: gavinkoma
"""

#%%
#start by importing the modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#%%
#start by generating 11 independent sets of random data
mu = 1 #mean
var = 1**2 #variance
deltrange = np.arange(0.9, 1.1, 0.02) #state mean values 0.9-1.1
output = [round(float(x), 2) for x in deltrange] #float the values to subtract
samples = [1, 5, 10**1, 50, 10**2, 500, 10**3,
           5000, 10**4, 50000, 10**5, 500000, 10**6] #13 sample sizes total


d = {} #initialize dictionary
for val in output:
    d["dataframe{0}".format(val)] = np.random.normal(
        loc=val, scale=var, size=1000000)

#%% q1 we need to start with mean 1
#and then perform the max likelihood
#use the log scale
for i,j in d.items(): #name, value
    mean = sum(j)/len(j) #calculate mean
    print("Mean of the dataset with delta value {val} is: {kval}".format(
        val=i, kval=mean)) #print statement for each value

#define the plot
mean_1 = d['dataframe1.0'] #pull mean 1 from dataframe
count = [(count+1,value) for count,value in enumerate(mean_1)] #count & keep track
mean_1 = pd.DataFrame(count,columns=['n','mean']) #make column names
mean_1['mean'].mean()#calc mean
mean1plot = mean_1.plot.scatter(x='n',y='mean')#plot
plt.ylabel('Mean Estimate')
plt.xlabel('N Values on a Log Base Scale')
plt.title('Estimate of Mean for range N = [1,10^6]')

#%%
#okay so we need to olook at the value for mean 1
#but we can actually just make a dict with them all 
#actually we cant bc theyll have different gauss vals

distributions = {}#here we have distribution dict
for i in samples:#we need all samp values but only with mean 1
    data = [random.gauss(1,var) for j in range(i)]
    distributions[i] = data
    
#calculate the mean
mean1 = []#define a list for the mean
for i,data in distributions.items():#name,val
    mean=sum(data)/len(data)#calc mean
    mean1.append(mean)#append

# question 2
plot = list(zip(samples, mean1))#tuples!!
df2 = pd.DataFrame(plot, columns=['n', 'mean'])#make a dataframe

fig, ax = plt.subplots()#plot
ax.scatter(samples,mean1, s=80, color='r')#plot
ax.set_xscale("Log")
plt.title('Plot of Mean Estimate (mu=1.00)')
plt.ylabel('Mean')
plt.xlabel('N')

#%%
#lets make a dictionary for them all now
dict_grand = {}
for i in output:
    dataset = {}
    for j in samples:
        key = "{}".format(j)
        value = [random.gauss(i, var) for k in range(j)]
        dataset[key] = value
    dict_grand["mean_{}".format(i)] = dataset
    
means_array = []
for dataset_name, dataset in dict_grand.items():
    for key, values in dataset.items():
        mean = np.mean(values)
        means_array.append((dataset_name, key, mean))

df_mean = pd.DataFrame(means_array, columns=['mean', 'n', 'estimate'])

# #%% god bless stack overflow
# all_distributions = {}
# for i in output: #these are the delta values
#     datasets = {} #empty
#     for j in samples: #n values
#         key = "{}".format(j) #naming system
#         #value = distribution of varying mean and var of 1 for
#         #all values in the range of the sample number
#         value = [random.gauss(i,var) for k in range(j)] 
#         dataset[key] = value #key value pair
#     all_distributions["mean_{}".format(i)] = dataset
    
# #%% define means for them
# mu_array = []
# for name,datasets in all_distributions.items(): #iterate name & values of main dict
#     for key, val in datasets.items(): #iterate values and key in the smaller dict
#         mean = np.mean(val)
#         mu_array.append((name,key,mean))

# all_means = pd.DataFrame(mu_array,columns=['mean','n','estimate'])

#%% we should plot them all:
mean_values = df_mean.iloc[:, 0].unique()

for i in mean_values:
    fig, ax = plt.subplots()
    plot_data = df_mean[df_mean["mean"] == i]
    ax.scatter(plot_data['n'],plot_data['estimate'], s=80, color='r')
    plt.title('Plot of Mean Estimate')
    plt.ylabel('{}'.format(i))
    plt.xlabel('Log10 Scale of N')
    ax.set_xscale('log')
    
#%% 
#compute the average of all of them now
#we already have a dict ==> df_mean

sorted_df = df_mean.sort_values(by=['n'], axis=0,kind = "mergesort")
mom = sorted_df.groupby('n')['estimate'].sum()/11
mom = mom.reset_index()

print(mom.head())

plt.scatter(mom['n'],mom['estimate'],)
plt.xscale('Log')
plt.ylabel('Average of Means')
plt.xlabel('N')
plt.title('Average of Mean Estimates')
plt.show()

#%%
#assume an initial mean guess of 2
init_mean = 2
var=1

#idk what the mean is supposed to be here...
true_mean = 1
bayes_estimations = []

for i in samples:
    posterior_mean = ((2)+i*true_mean)/(1/var+i)
    bayes_estimations.append(posterior_mean)

plt.plot(samples,bayes_estimations)
plt.xscale('Log')
plt.xlabel('n')
plt.ylabel('estimate')
plt.title('bayesian estimate of the mean')
plt.show()



