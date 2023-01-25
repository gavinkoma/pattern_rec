#!/usr/bin/env python
#
# file: /data/isip/exp/demos/exp_0007/v0.0.6/imld_alg_lda.py
#
# revision history:
# 20200811 (LV): standardization, completion
# 20200505 (SJ): initial version
#
# This script implements the Class Dependent(Quadratic) and Independent Linear
# Discriminant Analysis machine learning algorithm for the ISIP Machine
# Learning Demo software.
#-----------------------------------------------------------------------------

# import system modules
#
import numpy as np

# import modules
#
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# ------------------------------------------------------------------------------
#
# global variables are listed here
#
# ------------------------------------------------------------------------------
CI = "CI"
CD = "CD"
NUM_STEPS = 3
LDA_DIM = 1
FORMAT = "{:<15} {:<15}"
PARAMETER = "PARAMETER"
VALUE = "VALUE"

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

#  class: AlgorithmKMeans
#
#  This class contains methods to apply the LDA and QDA algorithm on a set of
#  data that will result in class classification, and both the computation and
#  printing of mean and covariance matrices.
#

class AlgorithmLDA():
    # method: AlgorithmLDA::constructor
    #
    # arguments:
    #  win_input: GUI input display
    #  win_output: GUI output display
    #  win_log: GUI process log
    #
    # return: none
    #
    def __init__(self, win_input, win_output, win_log, mode):

        # create class data
        #
        AlgorithmLDA.__CLASS_NAME__ = self.__class__.__name__

        # copy the inputs into class data
        #
        self.input_d = win_input
        self.output_d = win_output
        self.log_d = win_log
        self.mode = mode

        # exit gracefully
        #
        return None

    # method: AlgorithmLDA::initialize
    #
    # arguments:
    #  data: data recorded from display that will be used for training
    #
    # return:
    #  True
    #
    # initialize variables for LDA
    #
    def initialize(self,data):
        self.data = data
        self.cov_mat = None
        self.mean_mat = None
        self.trans_mat = None
        self.dist = None
        self.lda = None
        self.lda_classes = []
        self.classes = len(self.data)
        self.step_index = 0

        # exit gracefully
        #
        return True

    # method: AlgorithmLDA::run_algo
    #
    # arguments:
    #  data: data recorded from display
    #
    # return:
    #  True
    #
    # run algorithm steps
    #
    def run_algo(self,data):

        # calc everything and display stats
        self.initialize(data)
        self.compute_stats()
        self.display_means()
        self.display_cov()

        # exit gracefully
        #
        return True

    # method: AlgorithmLDA::create_labels
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
            labels = labels + [count]*total_samples
            count +=1

        # convert into array
        #
        labels = np.array(labels)

        # exit gracefully
        #
        return labels

 # method: AlgorithmLDA::compute_stats
    #
    # arguments: none
    #
    # return:
    #  True
    #
    # this method calculates the covariance and means matrix for each class
    # LDA is applied to
    #
    def compute_stats(self):

        # initialize variables
        #
        self.mean_mat = []
        self.cov_mat = []
        self.trans_mat = []
        self.data = np.array(self.data)
        self.labels = self.create_labels()

        # depending on either mode use either LDA or QDA
        #
        if self.mode == CI:
            self.lda = LDA(solver='eigen', store_covariance=True,
                           n_components=None)
        else:
            self.lda = QDA(store_covariance=True)

        data = np.vstack((self.data))
        self.lda.fit(data,self.labels)
        tmp_cov = np.array(self.lda.covariance_)
        self.cov_mat.append(tmp_cov)
        self.mean_mat = self.lda.means_
        self.print_params()

        # exit gracefully
        #
        return True

 # method: AlgorithmLDA::display_means
    #
    # arguments: none
    #
    # return:
    #  True
    #
    # this method display the means for each class and plots a point at the mean
    #
    def display_means(self):
        text = "\nMeans: \n"
        count = 0
        for line in self.mean_mat:
            text += "Class" + str(count) + ': ' + str(line) + '\n'
            self.input_d.canvas.axes.scatter(line[0], line[1], c='black', s=7)
            count += 1

        self.log_d.append(text)
        self.input_d.canvas.draw_idle()

        # exit gracefully
        #
        return True

  # method: AlgorithmLDA::display_cov
    #
    # arguments: none
    #
    # return:
    #  True
    #
    # this method displays the covariance matrix of each class in the process log
    #
    def display_cov(self):
        text = 'Covariance Matrix: \n'
        print(self.cov_mat)
        for classes in range(len(self.cov_mat)):
            text += 'Class' + str(classes) +': '
            str_cov = np.array2string(self.cov_mat[classes], formatter={'float_kind': lambda x: "%.4f" % x})
            text += '\n' + str_cov+ '\n'

        self.log_d.append(text)

        # exit gracefully
        #
        return True

    def print_params(self):
        param = self.lda.get_params()
        self.log_d.append("\n"+(FORMAT.format
                           (PARAMETER, VALUE)))

        for k, v in param.items():
            k, v = str(k), str(v)

            self.log_d.append(FORMAT.format(k, v))

        return True

    # method: AlgorithmLDA::predict
    #
    # arguments:
    #  ax: the canvas with the original data is plotted on
    #  X: is the data that is being used for the predictions
    #
    # return:
    #  xx: the x coordinates of the contour
    #  yy: the y coordinates of the contour
    #  Z: the height of the contour
    #
    # This method is used to make a prediction using the Mahalanobis distance
    #
    def predict(self, ax, X):

        # Creates the mesh grid
        #
        X = np.concatenate(X, axis=0)
        X = np.reshape(X, (-1, 2))
        res = (ax.canvas.axes.get_xlim()[1] - ax.canvas.axes.get_ylim()[0]) / 100
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))

        a = np.c_[xx.ravel(),yy.ravel()]
        Z = self.lda.predict(a)
        Z = Z.reshape(xx.shape)

        # exit gracefully
        #
        return xx, yy, Z

    # method: AlgorithmLDA::prediction_classifier
    #
    # arguments:
    #  data: the class data being used for predictions
    #
    # return:
    #  distance: minimal distance
    #
    # this method calculates the distance prediction of each class
    def prediction_classifier(self, data):

        # predict class label
        #
        data = np.vstack((data))
        prediction = self.lda.predict(data)

        # exit gracefully
        #
        return prediction

#
# end of class

#
# end of file