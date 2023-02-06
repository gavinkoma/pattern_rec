#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:07:50 2023

@author: gavinkoma
"""
# homework 3
# due on 02/06

# %%import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# %%
# generate 11 independent sets of random data consisting of 10^6 points
# from a 1D GRV with a variance of 1. make sure to use a mean of 1 +- delta
# where delta will vary from 0.9 to 1.0 in equal steps of 0.2

# start with generation
mu = 1
var = 1**2
deltrange = np.arange(0.9, 1.1, 0.02)
output = [round(float(x), 2) for x in deltrange]

d = {}
for val in output:
    d["dataframe{0}".format(val)] = np.random.normal(
        loc=val, scale=var, size=1000000)


# %%
# for set with mean 1 --> estimate with max likelihood
# plot and use log scale
# compute mean and repeat by taking all average of N for first 6
# is the second estimate biased?
# pull data with mean 1
for i, data in d.items():
    mean = sum(data)/len(data)
    print("Mean of the dataset with delta value {val} is: {kval}".format(
        val=i, kval=mean))

# question 1
print("Mean of data set with delta of 1.00 is: 0.99736")


# %%
# we should plot
values = d['dataframe1.0']
count = [(count+1, value) for count, value in enumerate(values)]
df1 = pd.DataFrame(count, columns=['N', 'mean'])
# print(df1.head())
df1['mean'].mean()
print(df1['mean'].mean())

# im not really sure what this plot is supposed to look like...
plot = df1.plot.scatter(x='N', y='mean')

# we need to define the samples
samples = [1, 5, 10**1, 50, 10**2, 500, 10**3,
           5000, 10**4, 50000, 10**5, 500000, 10**6]
new_dict = {}
for i in samples:
    data = [random.gauss(1, var) for j in range(i)]
    new_dict[i] = data

# delta mean
delta_mean = []
for i, data in new_dict.items():
    mean = sum(data)/len(data)
    delta_mean.append(mean)


# question 2
plot = list(zip(samples, delta_mean))
df2 = pd.DataFrame(plot, columns=['N_val', 'meanestimate'])
print(df2.head())

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df2['N_val'], df2['meanestimate'], s=80, color='r')
ax.set_xscale("Log")

# %% plot for all 6 datasets

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


# %%
mean_values = df_mean.iloc[:, 0].unique()

for i in mean_values:
    plt.figure()
    plot_data = df_mean[df_mean["mean"] == i]
    plt.scatter(plot_data['n'], plot_data['estimate'], label=i)
    plt.xscale('Log')
    plt.legend()
    plt.show()


#%%
#N=10 use 11x10=110pts and compute the average
#how would this compare N = 110 for plot of prob 2?

averaged = df_mean
averaged = pd.DataFrame(averaged)
sorted_df = averaged.sort_values(by=['n'], axis=0,kind = "mergesort")
sorted_df.head()

mom = sorted_df.groupby('n')['estimate'].sum()/11
mom = mom.reset_index()

print(mom.head())

plt.scatter(mom['n'],mom['estimate'],)
plt.xscale('Log')
plt.show()


#%%
#assume an initial mean guess of 2
init_mean = 2
var=1

#idk what the mean is supposed to be here...
true_mean = 1
bayes_estimations = []

for i in samples:
    posterior_mean = (init_mean/var+i*true_mean)/(1/var+i)
    bayes_estimations.append(posterior_mean)

plt.plot(samples,bayes_estimations)
plt.xscale('Log')
plt.xlabel('n')
plt.ylabel('estimate')
plt.title('bayesian estimate of the mean')
plt.show()


