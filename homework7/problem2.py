#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:36:33 2023

@author: gavinkoma
"""

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math

os.chdir(r'/Users/gavinkoma/Desktop/pattern_rec/homework7/data')
print(os.getcwd())

#area score = .1

ci = 0.95
p1 = 0.20
p2 = 0.1890
N = 10000

def z_score(ci,p1,p2,N):
    num = p1-p2
    A = (p1*(1-p1)/N)
    B = (p2*(1-p2)/N)
    denom = math.sqrt(A+B)
    #calc
    z = num/denom
    
    print("\ncomputed z-score: " + str(z))
    return z

def area_score(ci):
    z = (1-ci)/2
    print("\narea z-score: " + str(z))

z_score(ci,p1,p2,N)
area_score(ci)


