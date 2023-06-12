# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%import stuff
import pandas as pd
import numpy as np

#%% part 1
#define the values as per the homework assignment
classes = [*range(1,5)]

mean1 = [1,1]
mean2 = [1,-1]
mean3 = [-1,-1]
mean4 = [-1,1]

means = (mean1,mean2,mean3,mean4)#condense for ease

#define the diagonal covariance matrix that we will use later
cov = [[0.1,0], [0,0.1]]

#now i guess we should create the data samples
#use the covariance matrix to generate data
class1data = np.random.multivariate_normal(mean1, cov, size=100)
class2data = np.random.multivariate_normal(mean2, cov, size=100)
class3data = np.random.multivariate_normal(mean3, cov, size=100)
class4data = np.random.multivariate_normal(mean4, cov, size=100)

dfclass1 = pd.DataFrame(class1data,columns=['class1vectorx','class1vectory'])
dfclass2 = pd.DataFrame(class2data,columns=['class2vectorx','class2vectory'])
dfclass3 = pd.DataFrame(class3data,columns=['class3vectorx','class3vectory'])
dfclass4 = pd.DataFrame(class4data,columns=['class4vectorx','class4vectory'])

df_final = pd.concat([dfclass1,dfclass2],axis=1)
df_final = pd.concat([df_final,dfclass3],axis=1)
df_final = pd.concat([df_final,dfclass4],axis=1)

df_final.to_csv("data_part1.csv",header=True,index=False) #save to csv

#%%part 2
#increase covar
#define the diagonal covariance matrix that we will use later
cov_2 = [[1.0,0], [0,1.0]]

class1data = np.random.multivariate_normal(mean1, cov_2, size=100)
class2data = np.random.multivariate_normal(mean2, cov_2, size=100)
class3data = np.random.multivariate_normal(mean3, cov_2, size=100)
class4data = np.random.multivariate_normal(mean4, cov_2, size=100)

dfclass1 = pd.DataFrame(class1data,columns=['class1vectorx','class1vectory'])
dfclass2 = pd.DataFrame(class2data,columns=['class2vectorx','class2vectory'])
dfclass3 = pd.DataFrame(class3data,columns=['class3vectorx','class3vectory'])
dfclass4 = pd.DataFrame(class4data,columns=['class4vectorx','class4vectory'])

df_final = pd.concat([dfclass1,dfclass2],axis=1)
df_final = pd.concat([df_final,dfclass3],axis=1)
df_final = pd.concat([df_final,dfclass4],axis=1)

df_final.to_csv("data_part2.csv",header=True,index=False) #save to csv


df_final.to_csv("data_part2.csv",header=True,index=False) #save to csv