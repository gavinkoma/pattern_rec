#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:44:54 2023

@author: gavinkoma
"""
#%%
#import any libraries we need
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

#%% build model
#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

#reshaping data
X_train = X_train.reshape((X_train.shape[0], 
                           X_train.shape[1], 
                           X_train.shape[2], 1))

X_test = X_test.reshape((X_test.shape[0],
                         X_test.shape[1],
                         X_test.shape[2],1)) 

#checking the shape after reshaping
print(X_train.shape)
print(X_test.shape)

#normalizing the pixel values
X_train=X_train/255
X_test=X_test/255

#defining model
model=Sequential()
#adding convolution layer
model.add(Conv2D(32,(8,8),
                 activation='relu',
                 input_shape=(28,28,1)))

#adding pooling layer
model.add(MaxPool2D(4,4))

#adding fully connected layer
model.add(Flatten())
model.add(Dense(100,
                activation='relu'))

#adding output layer
model.add(Dense(10,
                activation='softmax'))

#compiling the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%%
#fitting the model
history = model.fit(X_train,y_train,
                    epochs=10,
                    verbose=1,
                    validation_data=(X_test,y_test))

#%%
error = []
for val in history.history['accuracy']:
    errorval = 1-val
    error.append(errorval)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.figure()
plt.plot(error)
plt.title("error rate")
plt.xlabel("epochs")
plt.ylabel("error rate")
plt.show()

#%% evaluate
model.evaluate(X_test,y_test)

