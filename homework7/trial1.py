#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:27:21 2023

@author: gavinkoma
"""

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework7/data')
print(os.getcwd())

dev = pd.read_csv('dev.csv',header=None)
evaluation = pd.read_csv('eval.csv',header=None)


print(len(dev))
print(len(evaluation))