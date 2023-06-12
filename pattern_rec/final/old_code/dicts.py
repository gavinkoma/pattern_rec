#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 19:16:38 2023

@author: gavinkoma
"""
import os 
import pandas as pd

os.chdir('/Users/gavinkoma/Desktop/pattern_rec/final/data_s14/')
input_dir = '/Users/gavinkoma/Desktop/pattern_rec/final/data_s14/'

folders = ['/train/', '/dev/']
# start with just 1 dev folder
dev_sub = ['10/']  # ,'11/']
train_sub = ['00/']

# start with just one train folder
# for i in range(0,10):
#    file = '0'+str(i)+str('/')
#    train_sub.append(file)


# create fucntion which will read in data into matrix style format

def matrix_data(filepath):

    # read data in, split data to just get event length data and remove
    # unneccessary strings, columns and rename/change type of columns

    df = pd.read_csv(filepath, header=None, skiprows=3)
    n_row = df[df.iloc[:, 0].eq("#------------")].index[0]
    df = pd.read_csv(filepath, header=None, skiprows=3, nrows=n_row)
    df = df.apply(lambda x: x.str.replace(
        'class:', '').str.replace('stop =', ''))
    df[0] = df[0].str.replace(':', '').str.split().str[-1].astype(int)
    df[1] = df[1].str.replace('start: ', '').str.split().str[-1]
    df['dur'] = df[2].str.split().str[-1].astype(int)
    df.drop(columns=2, inplace=True)
    df[0] = df[0].astype(int)
    df[1] = df[1].astype(int)
    df['dur'] = df['dur'].astype(int)

    # create a list which measures the length of the event
    # and clear the uneccesary columns

    diff_list = []
    for i in range(len(df)):
        diff = df.iloc[i]['dur'] - df.iloc[i][1] + 1
        diff_list.append(diff)

    df['diff'] = diff_list
    df.drop([1, 'dur'], axis=1, inplace=True)
    df['class'] = df[0]
    df['length'] = df['diff']
    df.drop([0, 'diff'], axis=1, inplace=True)

    # extract class and length values and to append these into a list and then
    # concatenate into a df, which we will drop length column from& reset index

    new_rows = []
    for i, row in df.iterrows():

        class_val = row['class']
        length_val = row['length']

        new_df = pd.DataFrame({
            'class': [class_val] * length_val,
            'length': [1] * length_val
        })

        new_rows.append(new_df)

    new_df = pd.concat(new_rows)
    new_df = new_df.reset_index(drop=True)
    new_df = new_df.drop(columns='length', axis=1)

    # read raw audio values and concat this with the event class values
    # to create our dfs
    df2 = pd.read_csv(filepath, header=None, skiprows=n_row+4)
    df2 = df2.drop(columns=0, axis=1)
    df = pd.concat([new_df, df2], axis=1)

    return df


# create folders to append into
train_dict = {}
dev_dict = {}


# set up for loop to create all different dataframes
# first loop calls either dev or train
# second loop calls all sub folders in these directories
# final loop calls all file names in the directories and
# makes use of the function created above to turn the
# csv file into desired dataframe
# set up iteration count as this takes quite a while

count = 0

for i, folder in enumerate(folders):

    if folder == '/train/':
        sub_file = train_sub

    elif folder == '/dev/':

        sub_file = dev_sub

    for j, sub in enumerate(sub_file):

        filepath = input_dir+folder+sub
        files = os.listdir(filepath)

        for k, file in enumerate(files):

            if folder == '/train/':
                train_dict[file] = matrix_data(filepath+file)

            if folder == '/dev/':
                dev_dict[file] = matrix_data(filepath+file)

            count += 1
            print("Iteration", count, ":", i, j, k)

        # check to see if above worked

if len(train_dict) == 1000 & len(dev_dict) == 1000:
    print("wow it actually worked")
else:
    print('try again :(')
    
    
#okay but like now what do we do
#im not really sure how to progress now bc we need to implement an encoder/decoder





































