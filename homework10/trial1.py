#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:55:30 2023

@author: gavinkoma
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
import seaborn as sns

os.chdir("/Users/gavinkoma/Desktop/pattern_rec/homework10/data")


traindata = pd.read_csv("train_03.csv",header=None)
devdata = pd.read_csv("dev_03.csv",header=None)
evaldata = pd.read_csv("eval_03.csv",header=None)


#%%we need to run randomforest first
#before using pca we whould train the model and use a logistic
#regression to just see how well it performs
x_train = traindata.iloc[:,1:3]
y_train = traindata.iloc[:,0]
x_dev = devdata.iloc[:,1:3]
y_dev = devdata.iloc[:,0]
x_eval = evaldata.iloc[:,1:3]
y_eval = evaldata.iloc[:,0]

plt.figure()
sns.scatterplot(x=traindata.iloc[:,1],
                y=traindata.iloc[:,2],
                hue = traindata.iloc[:,0]
                ).set(title="Training Data (D10)")



for val in np.arange(1,11,1):    
    rf = RandomForestClassifier(criterion = 'gini',
                                n_estimators=val,
                                random_state=45,
                                n_jobs=(None))
    rf.fit(x_train,y_train)

    y_pred = rf.predict(x_train)
    accuracy = accuracy_score(y_train,y_pred)
    print("Accuracy with {} tree(s): ".format(val),accuracy)

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    
    #train data
    fig,ax = plt.subplots()
    plot_decision_regions(x_train, y_train,clf=rf)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('RNF Decision Boundary with {} Tree(s)'.format(val))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    x_train = traindata.iloc[:,1:3]
    y_train = traindata.iloc[:,0]


#%% dev & eval & train
x_train = traindata.iloc[:,1:3]
y_train = traindata.iloc[:,0]
x_dev = devdata.iloc[:,1:3]
y_dev = devdata.iloc[:,0]
x_eval = evaldata.iloc[:,1:3]
y_eval = evaldata.iloc[:,0]

rf = RandomForestClassifier(criterion = 'gini',
                            n_estimators=7,
                            random_state=45,
                            n_jobs=(None))
rf.fit(x_train,y_train)

y_pred = rf.predict(x_train)
accuracy = accuracy_score(y_train,y_pred)
print("Accuracy with 7 tree(s) for train: ",accuracy)

y_pred = rf.predict(x_dev)
accuracy = accuracy_score(y_dev,y_pred)
print("Accuracy with 7 tree(s) for dev: ",accuracy)

y_pred = rf.predict(x_eval)
accuracy = accuracy_score(y_eval,y_pred)
print("Accuracy with 7 tree(s) for eval: ",accuracy)


#%%we want to plot the performance of the eval set not dev set
x_eval = evaldata.iloc[:,1:3]
y_eval = evaldata.iloc[:,0]

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

#it doesnt seem like we are able to get higher than like 66% accuracy

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





#%%do the SVM
#this is too big of a dataset for a kernal svm sooo lets delete like 90% of our data
train_shuffle = traindata.sample(frac=1)
train_10 = train_shuffle.iloc[0:10000,:]
x_10_train = train_10.iloc[:,1:3]
y_10_train = train_10.iloc[:,0]

clf = svm.SVC(kernel="linear") #linear kernel
clf.fit(x_10_train,y_10_train)
y_pred = clf.predict(x_eval)
print("accuracy of Linear SVM: ",metrics.accuracy_score(y_eval,y_pred))

for val in np.arange(1,10,1):
    svclassifier = svm.SVC(kernel='poly', degree = val)
    svclassifier.fit(x_10_train,y_10_train)
    y_pred_poly = svclassifier.predict(x_eval)
    accuracy = accuracy_score(y_eval,y_pred_poly)
    print("accuracy for Poly SVM of {} degree".format(val),accuracy)

svclassifier = svm.SVC(kernel='rbf')
svclassifier.fit(x_10_train,y_10_train)
y_pred_rbf = svclassifier.predict(x_eval)
accuracy = accuracy_score(y_eval,y_pred_rbf)
print("accuracy of RBF SVM: ",accuracy)

svclassifier = svm.SVC(kernel='sigmoid')
svclassifier.fit(x_10_train,y_10_train)
y_pred_sig = svclassifier.predict(x_eval)
accuracy = accuracy_score(y_eval,y_pred_sig)
print("accuracy of Sigmoidal SVM: ",accuracy)

#%%
x_train = traindata.iloc[:,1:3]
y_train = traindata.iloc[:,0]
x_dev = devdata.iloc[:,1:3]
y_dev = devdata.iloc[:,0]
x_eval = evaldata.iloc[:,1:3]
y_eval = evaldata.iloc[:,0]

svclassifier = svm.SVC(kernel='rbf')
svclassifier.fit(x_10_train,y_10_train)

y_pred_rbf = svclassifier.predict(x_train)
accuracy = accuracy_score(y_train,y_pred_rbf)
print("accuracy of RBF SVM with train: ",accuracy)

y_pred_rbf = svclassifier.predict(x_dev)
accuracy = accuracy_score(y_dev,y_pred_rbf)
print("accuracy of RBF SVM with dev: ",accuracy)

y_pred_rbf = svclassifier.predict(x_eval)
accuracy = accuracy_score(y_eval,y_pred_rbf)
print("accuracy of RBF SVM with eval: ",accuracy)



#%% part 2
traindata = pd.read_csv("train_03.csv",header=None)
devdata = pd.read_csv("dev_03.csv",header=None)
evaldata = pd.read_csv("eval_03.csv",header=None)

#before using pca we whould train the model and use a logistic
#regression to just see how well it performs
x_train = traindata.iloc[:,1:3]
y_train = traindata.iloc[:,0]
x_eval = evaldata.iloc[:,1:3]
y_eval = evaldata.iloc[:,0]


# Fit QDA to the training data
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train, y_train)
# Make predictions on the test data
y_pred_qda = qda.predict(x_eval)
# Calculate the FPR and TPR at different threshold values
fpr, tpr, thresholds = roc_curve(y_eval, y_pred_qda)
# Calculate the AUC score
auc_qda = roc_auc_score(y_eval, y_pred_qda)


#this is too big of a dataset for a kernal svm sooo lets delete #like 90% of our data
train_shuffle = traindata.sample(frac=1)
train_10 = train_shuffle.iloc[0:10000,:]
x_10_train = train_10.iloc[:,1:3]
y_10_train = train_10.iloc[:,0]

clf = svm.SVC(kernel="linear") #linear kernel
clf.fit(x_10_train,y_10_train)
y_pred_svm = clf.predict(x_eval)
fpr_svm,tpr_svm,threshold_svm = roc_curve(y_eval,y_pred_svm)
auc_svm = roc_auc_score(y_eval, y_pred_svm)



# Fit RF to the training data
rf = RandomForestClassifier(criterion = 'gini',
                            n_estimators=100,
                            random_state=45,
                            n_jobs=(None))
rf.fit(x_train, y_train)
# Make predictions on the test data
y_pred_rf = rf.predict(x_eval)
# Calculate the FPR and TPR at different threshold values
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_eval, y_pred_rf)
# Calculate the AUC score
auc_rf = roc_auc_score(y_eval, y_pred_rf)


# Plot the ROC curve
plt.figure()
plt.plot(fpr_rf, tpr_rf, label='RF (AUC = %0.2f)' % auc_rf)
plt.plot(fpr_svm, tpr_svm, label='SVM (AUC = %0.2f)' % auc_svm)
plt.plot(fpr, tpr, label='QDA (AUC = %0.2f)' % auc_qda)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


#%%we need the auc values too
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from sklearn import svm

x_train = traindata.iloc[:,1:3]
y_train = traindata.iloc[:,0]
x_dev = devdata.iloc[:,1:3]
y_dev = devdata.iloc[:,0]
x_eval = evaldata.iloc[:,1:3]
y_eval = evaldata.iloc[:,0]

# train QDA model on training data
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train, y_train)
# predict probabilities for test data
probas = qda.predict_proba(x_eval)
# compute FPR, TPR, and threshold values for ROC curve
fpr, tpr, thresholds = roc_curve(y_eval, probas[:, 1], pos_label=1)
# compute AUC
roc_auc = auc(fpr, tpr)
print("AUC values for qda: ",roc_auc)



# train Random Forest model on training data
rf_model = RandomForestClassifier(criterion = 'gini',
                            n_estimators=7,
                            random_state=45,
                            n_jobs=(None))

rf_model.fit(x_train, y_train)
# predict probabilities for test data
probas = rf_model.predict_proba(x_eval)
# compute FPR, TPR, and threshold values for ROC curve
fpr, tpr, thresholds = roc_curve(y_eval, probas[:, 1], pos_label=1)
# compute AUC
roc_auc = auc(fpr, tpr)
print("accuracy for RNF: ", roc_auc)



#this is too big of a dataset for a kernal svm sooo lets delete #like 90% of our data
train_shuffle = traindata.sample(frac=1)
train_10 = train_shuffle.iloc[0:10000,:]
x_10_train = train_10.iloc[:,1:3]
y_10_train = train_10.iloc[:,0]

# train SVM model on training data
svm_model = svm.SVC(probability=True)
svm_model.fit(x_train, y_train)
# predict probabilities for test data
probas = svm_model.predict_proba(x_eval)
# compute FPR, TPR, and threshold values for ROC curve
fpr, tpr, thresholds = roc_curve(y_eval, probas[:, 1], pos_label=1)
# compute AUC
roc_auc = auc(fpr, tpr)
print("accuracy for SMV: ",roc_auc)

#%% # of support vectors increasing
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

x_train = traindata.iloc[:,1:3]
y_train = traindata.iloc[:,0]
x_dev = devdata.iloc[:,1:3]
y_dev = devdata.iloc[:,0]
x_eval = evaldata.iloc[:,1:3]
y_eval = evaldata.iloc[:,0]

#this is too big of a dataset for a kernal svm sooo lets delete #like 90% of our data
train_shuffle = traindata.sample(frac=1)
train_10 = train_shuffle.iloc[0:10000,:]
x_10_train = train_10.iloc[:,1:3]
y_10_train = train_10.iloc[:,0]
x_train = x_10_train
y_train = y_10_train


svm_accuracy = []
svm_vectors = []

for val in np.arange(1,50,1):
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(x_train[val:],y_train[val:])
    y_pred = svm.predict(x_eval)
    accuracy = accuracy_score(y_eval,y_pred)
    print("accuracy with {} vectors: ".format(val),accuracy)
    svm_accuracy.append(accuracy)
    svm_vectors.append(val)

# Plot the accuracy as a function of the number of support vectors
plt.plot(svm_vectors, svm_accuracy)
plt.xlabel('Number of support vectors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Support Vectors')
plt.show()







