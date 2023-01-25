#!/usr/bin/env python
#
# file: imld/alg/imld_alg_mlp.py
#
# revision history:
#
# 20220210 (MM): initial version
# This script implements Multi Layer Perceptron machine learning algorithm for
# the ISIP Machine Learning Demo software.
#
#------------------------------------------------------------------------------
#
# imports are listed here
#
#------------------------------------------------------------------------------

# import modules
#
import numpy as np
from sklearn.neural_network import MLPClassifier
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

#  class: AlgorithmMLP
#
#  This class contains methods to apply the MLP algorithm on a set of data
#  with a choice on numbers of layers and solver.
#

class AlgorithmMLP():
    # method: AlgorithmMLP::constructor
    #
    # arguments:
    #  win_input: GUI input display
    #  win_output: GUI output display
    #  win_log: GUI process log
    #  n_layers: number of hidden layers
    #  solver: method choice between lbfgs, sgd, and adam
    #
    # return: none
    #
    def __init__(self, win_input, win_output, win_log, nlayers, solver):
        # create class data
        #
        AlgorithmMLP.__CLASS_NAME__ = self.__class__.__name__

        # copy the inputs into class data
        #
        self.input_d = win_input
        self.output_d = win_output
        self.log_d = win_log
        self.nlayers = nlayers
        self.solver = solver

        # exit gracefully
        #
        return None

    # method: AlgorithmMLP::initialize
    #
    # arguments:
    #  data: data recorded from display that will be used for training
    #
    # return:
    #  True
    #
    # initialize variables for MLP
    #
    def initialize(self, data):
        # initialize variables
        #
        self.data = data
        self.classes = len(self.data)
        self.mlp = MLPClassifier(hidden_layer_sizes=self.nlayers,
                                 solver=self.solver,random_state=icf.SEED)
        self.X = np.empty((0, 0))
        self.print_params()

        # exit gracefully
        #
        return True

    # method: AlgorithmMLP::run_algo
    #
    # arguments:
    #  data: data recorded from display
    #
    # return: True
    #
    # run algorithm steps
    #
    def run_algo(self, data):
        # initialize and train data
        self.initialize(data)
        self.train()

        # exit gracefully
        #
        return True

    # method: AlgorithmMLP::create_labels
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

        # set up lists of labels
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

        # convert into an array
        #
        labels = np.array(labels)

        # exit gracefully
        #
        return labels

    # method: AlgorithmMLP::train
    #
    # arguments:
    #  None
    #
    # return:
    #  True
    #
    # This method trains the model based on the data provided and labels made
    #
    def train(self):

        # create labels
        #
        labels = self.create_labels()

        # stack data and train algorithm
        #
        data = np.vstack((self.data))
        self.mlp.fit(data,labels)

        # exit gracefully
        #
        return True

    def print_params(self):
        param = self.mlp.get_params()
        self.log_d.append("\n"+(FORMAT.format
                           (PARAMETER, VALUE)))

        for k, v in param.items():
            k, v = str(k), str(v)

            self.log_d.append(FORMAT.format(k, v))
        return True

    # method: AlgorithmMLP::predict
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
        Z = self.mlp.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # exit gracefully
        #
        return xx, yy, Z

    # method: AlgorithmMLP::prediction_classifier
    #
    # arguments:
    #  data: data from the evaluation display
    #
    # return:
    #  prediction: the predicated class label
    #
    # This method predicts the class label
    #
    def prediction_classifier(self, data):

        data = np.vstack((data))
        prediction = self.mlp.predict(data)

        # exit gracefully
        #
        return prediction

#
# end of class

#
# end of file