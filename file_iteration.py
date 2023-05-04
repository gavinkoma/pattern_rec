#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:46:32 2023

@author: gavinkoma
"""
import os
import random

print('Loop over dirs and files:')
train_path = '/Users/gavinkoma/Desktop/pattern_rec/final/subset_data/'
dev_path = '/Users/gavinkoma/Desktop/pattern_rec/final/data_s14/dev/'
file_names_train = []
file_names_dev = []

for root, dirs, files in os.walk(train_path):
    #print(root)
    for _file in files:
        print(str(root)+str(_file))
        file_names_train.append(str(root)+str(_file))
                             
                             
for root, dirs, files in os.walk(dev_path):
    #print(root)
    for _file in files:
        print(str(root)+str(_file))
        file_names_dev.append(str(root)+str(_file))


#for file in training_sets:
#    df = process_data(file)

           
           









