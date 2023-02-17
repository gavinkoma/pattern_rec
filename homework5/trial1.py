#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:35:02 2023

@author: gavinkoma
"""

import numpy as np
import os
os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework5/')
print(os.getcwd())

#%%

#read text file
with open('myprog_file.txt') as f:
    lines = f.readlines()

#lowercase them
R = lines[0].lower()
R = R.split(" ")
H = lines[0].lower()
H = H.split(" ")



#parse into vectors



