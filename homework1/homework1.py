# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd


#define the values as per the homework assignment
classes = {'classes': ['class1','class2','class3','class4']}

df = pd.DataFrame(classes)

xval = [1,1,-1,-1]
yval = [1,-1,-1,1]

df['xvector']=xval
df['yvector']=yval
print(df.cov())

#define the diagonal covariance matrix that we will use later
cov = [[0.1,0], [0,0.1]]

#we need this data to go into jmp so export as csv
df.to_csv("initial_data.csv",header=True,index=False)