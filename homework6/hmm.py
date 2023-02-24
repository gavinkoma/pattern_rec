#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:20:14 2023

@author: gavinkoma
"""
#%%import and chdir
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from hmmlearn import hmm

#%% get data
#dont need to change the wd for this one! woo!

base_dir = "https://github.com/natsunoyuki/Data_Science/blob/master/gold/gold/gold_price_usd.csv?raw=True"

data = pd.read_csv(base_dir)
data["datetime"] = pd.to_datetime(data["datetime"])
data["gold_price_change"] = data["gold_price_usd"].diff()
data = data[data["datetime"] >= pd.to_datetime("2008-01-01")]

plt.figure(figsize = (15, 10))
plt.subplot(2,1,1)
plt.plot(data["datetime"], data["gold_price_usd"])
plt.xlabel("datetime")
plt.ylabel("gold price (usd)")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(data["datetime"], data["gold_price_change"])
plt.xlabel("datetime")
plt.ylabel("gold price change (usd)")
plt.grid(True)
plt.show()

#%%fit the daily change to ge model with 3 hidden states
X = data[["gold_price_change"]].values
model = hmm.GaussianHMM(n_components = 3, covariance_type = "diag", 
                        n_iter = 50, random_state = 42)
model.fit(X)

Z = model.predict(X)
states = pd.unique(Z)

print("Unique states:")
print(states)

print("\nStart probabilities:")
print(model.startprob_) #highest prob is found in state 0

print("\nTransition matrix:")
print(model.transmat_)

print("\nGaussian distribution means:")
print(model.means_)

print("\nGaussian distribution covariances:")
print(model.covars_)

#%% and now we want to plot
plt.figure(figsize = (15, 10))
plt.subplot(2,1,1)
for i in states:
    want = (Z == i)
    x = data["datetime"].iloc[want]
    y = data["gold_price_usd"].iloc[want]
    plt.plot(x, y, '.')
plt.legend(states, fontsize=16)
plt.grid(True)
plt.xlabel("datetime", fontsize=16)
plt.ylabel("gold price (usd)", fontsize=16)
plt.subplot(2,1,2)
for i in states:
    want = (Z == i)
    x = data["datetime"].iloc[want]
    y = data["gold_price_change"].iloc[want]
    plt.plot(x, y, '.')
plt.legend(states, fontsize=16)
plt.grid(True)
plt.xlabel("datetime", fontsize=16)
plt.ylabel("gold price change (usd)", fontsize=16)
plt.show()

