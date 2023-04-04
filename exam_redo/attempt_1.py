#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#going to just place hold

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

os.chdir("/Users/gavinkoma/Desktop/pattern_rec/exam_redo")

#%%question 1a
#okay lets recreate these data values
values = pd.DataFrame([[0.0, 0.50, 0],
                       [-0.75,0.00,0],
                       [-0.25, 0.00, 0],
                       [0.25,0.00,0],
                       [0.75,0.00,0],
                       [-0.50,0,1],
                       [0.0,0.0,1],
                       [0.50,0.0,1],
                       [0,-0.50,1]])

x0 = values.iloc[:,0]
x1 = values.iloc[:,1]
y = values.iloc[:,2]
X = values.iloc[:,0:2]

plt.scatter(values[0],values[1],c=values[2])
plt.xlim(-1,1)
plt.ylim(-1,1)

n_neighbors = 3

clf = neighbors.KNeighborsClassifier(n_neighbors,weights='uniform')
clf.fit(X,y)

cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']

PAD = 1.0 # how much to "pad" around the min/max values. helps in setting bounds of plot

x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

h=0.2

xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                    np.arange(y_min,y_max,h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, cmap=cmap_light)

sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1],hue=y,
                palette=cmap_bold, alpha=1.0, edgecolor="black")

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("3-Class classification (k = %i, 'uniform' = '%s')"
% (n_neighbors, 'uniform'))

plt.show()

#%%question 1b
#okay so we need to run a random forest and also plot the dec. boundary again
#err maybe we dont need to plot the boundary and we can just verify values

randomf_val = pd.read_excel("q2.xlsx",header = None)

plt.figure()
plt.scatter(randomf_val.iloc[:,0],randomf_val.iloc[:,1],c=randomf_val.iloc[:,2])
plt.xlim(-1,1)
plt.ylim(-1,1)


test_values = randomf_val.iloc[:,0:2]
test_features = randomf_val.iloc[:,2]

forest = RandomForestClassifier(criterion = 'gini',
                                n_estimators=3,
                                random_state=45,
                                n_jobs=(None))

forest.fit(test_values,test_features)

test_values = test_values.to_numpy()
test_features = test_features.to_numpy()

fig,ax = plt.subplots()
plot_decision_regions(test_values, test_features,clf=forest)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()





#%%we want to plot the performance of the eval set not dev set
x_eval = devdata.iloc[:,1:3]
y_eval = devdata.iloc[:,0]

rnf_accuracy = []
decision_trees = []
for val in np.arange(1,50,1):
    rf = RandomForestClassifier(criterion = 'gini',
                                n_estimators=val,
                                random_state=45,
                                n_jobs=(None))
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_eval)
    accuracy = accuracy_score(y_eval,y_pred)
    print("accuracy with {} tree(s): ".format(val),accuracy)
    rnf_accuracy.append(accuracy)
    decision_trees.append(val)
    
plt.figure()
sns.color_palette("pastel")
sns.lineplot(x=decision_trees,
             y=rnf_accuracy).set(title="Accuracy as a Function of Decision Trees",
                                 xlabel="Number of Decision Trees",
                                 ylabel="Accuracy Rate")
 
plt.show()





