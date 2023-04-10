#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:21:01 2023

@author: gavinkoma
"""


#%%import libraries
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch

#%%import data and do very basic graphs
os.chdir("/Users/gavinkoma/Desktop/pattern_rec/homework11/data")
train_data = pd.read_csv("train_03.csv",header=None)
dev_data = pd.read_csv("dev_03.csv",header=None)
eval_data = pd.read_csv("eval_03.csv",header=None)

plt.figure()
sns.scatterplot(x=train_data.iloc[:,1],
                y=train_data.iloc[:,2],
                data=train_data,
                hue=train_data.iloc[:,0]).set_title("train_data")


plt.figure()
sns.scatterplot(x=dev_data.iloc[:,1],
                y=dev_data.iloc[:,2],
                data=dev_data,
                hue=dev_data.iloc[:,0]).set_title("dev_data")


plt.figure()
sns.scatterplot(x=eval_data.iloc[:,1],
                y=eval_data.iloc[:,2],
                data=eval_data,
                hue=eval_data.iloc[:,0]).set_title("eval_data")


#%%implementation of SLP
x = torch.ones(1,requires_grad=True)
print(x.grad) #returns None initially
y=x+2
z=y*y*2
z.backward()
print(x.grad) #this is partial z/x


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, 
                                   self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output
    
#%%prep the datasets for training
x_train = train_data.iloc[:,1:3]
y_train = train_data.iloc[:,0]
x_dev = dev_data.iloc[:,1:3]
y_dev = dev_data.iloc[:,0]
x_eval = eval_data.iloc[:,1:3]
y_eval = eval_data.iloc[:,0]

x_train = torch.Tensor(x_train.values) #convert to an array 
x_dev = torch.Tensor(x_dev.values)
x_eval = torch.Tensor(x_eval.values) #array

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_dev = torch.FloatTensor(x_dev)
y_dev = torch.FloatTensor(y_dev)
x_eval = torch.FloatTensor(x_eval)
y_eval = torch.FloatTensor(y_eval)

#%%implement the model w/ tensors
model = Feedforward(2,100)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), 
                            lr = 0.01)

#%%evaluate before training
model.eval()
y_pred = model(x_eval)
before_train = criterion(y_pred.squeeze(),y_eval)
print('test loss before training',before_train.item())

#%%evluate after training
model.train()
epoch = 50

epoch_val = []
loss_val = []

for epoch in range(epoch):
    optimizer.zero_grad()
    
    #forward pass
    y_pred = model(x_train)
    
    #loss computation
    loss = criterion(y_pred.squeeze(),y_train)
    epoch_val.append(epoch)
    loss_val.append(loss.detach())
    print('Epoch: {} \n Train Loss: {} \n'.format(epoch,loss.item()))
    
    #backward pass
    loss.backward()
    optimizer.step()

plt.figure()
sns.scatterplot(x = epoch_val,y=loss_val).set_title("Loss vs. Epoch #")
plt.xlabel("Epoch #")
plt.ylabel("Loss Value")

#%%evaluation
model.eval()
y_pred = model(x_eval)
after_train = criterion(y_pred.squeeze(),y_eval)
print('Test loss after training: ',after_train.item())
    
#%%accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

accuracy_train = accuracy(x_train,y_train)
accuracy_eval = accuracy(y_pred,y_eval)

print("Error of Train: ", str(1-accuracy_train))
print("Error of Eval: ", str(1-accuracy_eval))





