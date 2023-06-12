#!/usr/bin/env python
#
# file: imld/alg/imld_alg_knn.py
#
# revision history:
#
# 20220210 (MM): initial version
# This script implements the Support Vector Machine (SVM) machine learning
# algorithm for the ISIP Machine Learning Demo software.
#
# ------------------------------------------------------------------------------
#
# imports are listed here
#
# ------------------------------------------------------------------------------

# import modules
#
import numpy as np
from sklearn.svm import SVC
import lib.imld_constants_file as icf

# ------------------------------------------------------------------------------
#
# global variables are listed here
#
# ------------------------------------------------------------------------------
FORMAT = "{:<15} {:<15}"
PARAMETER = "PARAMETER"
VALUE = "VALUE"
#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

#  class: AlgorithmSVM
#
#  This class contains methods to apply the SVM algorithm on a set of data
#  with a choice on max iterations and gamma value.
#

class AlgorithmSVM():
    # method: AlgorithmSVM::constructor
    #
    # arguments:
    #  win_input: GUI input display
    #  win_output: GUI output display
    #  win_log: GUI process log
    #  maxiter: number of maximum iterations
    #  gamma: algorithm choice of either auto or scale
    #
    # return: none
    #
    def __init__(self, win_input, win_output, win_log, maxiter, gamma):
        # create class data
        #
        AlgorithmSVM.__CLASS_NAME__ = self.__class__.__name__

        # copy the inputs into class data
        #
        self.input_d = win_input
        self.output_d = win_output
        self.log_d = win_log
        self.gamma = gamma
        self.maxiter = maxiter

        # exit gracefully
        #
        return None

    # method: AlgorithmSVM::initialize
    #
    # arguments: None
    #
    # return: True
    #
    # initialize variables for SVM
    #
    def initialize(self, data):

        # initialize variables
        #
        self.data = data
        self.classes = len(self.data)

        # set up SVM model
        #
        self.svm = SVC(gamma=self.gamma, max_iter=self.maxiter,
                       random_state=icf.SEED)

        self.X = np.empty((0,0))
        self.print_params()

        # exit gracefully
        #
        return True

    # method: AlgorithmSVM::run_algo
    #
    # arguments: None
    #
    # return: True
    #
    # run algorithm steps
    #
    def run_algo(self, data):

        # initialize and train algorithm
        #
        self.initialize(data)
        self.compute_vectors()
        self.draw_vectors()

        # exit gracefully
        #
        return True

    # method: AlgorithmSVM::create_labels
    #
    # arguments:
    #  None
    #
    # return:
    #  labels: labels for each data sample representing its class
    #
    # This method creates labels for the training data
    #
    def create_labels(self):

        # set up list of labels
        #
        labels = []
        count = 0

        # for each class append a samples length amount of labels
        #
        d = self.input_d.class_info
        for i in d:
            total_samples = len(d[i][1])
            labels = labels + [count] * total_samples
            count += 1

        # convert into array
        #
        labels = np.array(labels)

        # exit gracefully
        #
        return labels

    # method: AlgorithmSVM::compute_vectors
    #
    # arguments:
    #  None
    #
    # return:
    #  True
    #
    # This method computes the vectors for the SVM model
    #
    def compute_vectors(self):
        labels = self.create_labels()
        data = np.vstack((self.data))
        self.svm.fit(data,labels)

        self.support_vectors = self.svm.support_vectors_

        # exit gracefully
        #
        return True

    # method: AlgorithmSVM::draw_vectors
    #
    # arguments:
    #  None
    #
    # return:
    #  True
    #
    # This method plots the support vector machines
    #
    def draw_vectors(self):
        self.input_d.canvas.axes.scatter(self.support_vectors[:,0],
                                         self.support_vectors[:,1],
                                         facecolors='none',edgecolor='black',
                                         s=8)

        # exit gracefully
        #
        return True

    # method: AlgorithmKNN::print_params
    #
    # arguments:
    #  None
    #
    # return:
    #  True
    #
    # This method displays the parameters used within a model
    def print_params(self):

        # get parameters
        #
        param = self.svm.get_params()

        # print column headers
        #
        self.log_d.append("\n" + (FORMAT.format
                                  (PARAMETER, VALUE)))

        # print parameter and corresponding values
        #
        for k, v in param.items():
            k, v = str(k), str(v)
            self.log_d.append(FORMAT.format(k, v))

        # exit gracefully
        #
        return True

    # method: AlgorithmSVM::predict
    #
    # arguments:
    #  ax: display where  will be plotted
    #  X: data recorded from display
    #
    # return:
    #  xx: vector of X coordinates of a coordinate matrix
    #  yy: vector of Y coordinates of a coordinate matrix
    #  Z: prediction based on the coordinate matrix
    #
    # This method calculates the predictions used for a decision surface
    #
    def predict(self, ax, X):

        X = np.concatenate(X, axis=0)
        X = np.reshape(X, (-1, 2))
        res = (ax.canvas.axes.get_xlim()[1] -
               ax.canvas.axes.get_ylim()[0]) / 100

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))
        Z = self.svm.predict(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)

        # exit gracefully
        #
        return xx, yy, Z

    # method: AlgorithmSVM::prediction_classifier
    #
    # arguments:
    #  data: data from the evaluation display
    #
    # return:
    #  prediction: the predicated class label
    #
    # This method predicts the class label
    #
    def prediction_classifier(self,data):

        # predict class label
        #
        data = np.vstack((data))
        prediction = self.svm.predict(data)

        # exit gracefully
        #
        return prediction

#
# end of class

#
# end of file
