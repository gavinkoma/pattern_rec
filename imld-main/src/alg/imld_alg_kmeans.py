#!/usr/bin/env python
#
# file: imld/alg/imld_alg_kmeans.py
#
# revision history:
#
# 20220210 (MM): initial version
#
# This script implements KMeans machine learning algorithm for the ISIP Machine
# Learning Demo software.
#
#------------------------------------------------------------------------------
#
# imports are listed here
#
#------------------------------------------------------------------------------

# import modules
#
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
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

#  class: AlgorithmKMeans
#
#  This class contains methods to apply the KMeans algorithm on a set of data
#  with a choice on numbers of clusters, iterations and max run time.
#

class AlgorithmKMeans():
    # method: AlgorithmKMeans::constructor
    #
    # arguments:
    #  win_input: GUI input display
    #  win_output: GUI output display
    #  win_log: GUI process log
    #  n_cluster: number of clusters
    #  init: choice of method between Kmeans++ or Random
    #  n_init: number of times Kmeans is ran to determine the best centroid seed
    #  maxiter: max number of iterations for a single run
    #
    # return: none
    #
    def __init__(self, win_input, win_output, win_log, n_clusters, init, n_init,
                 maxiter):
        # create class data
        #
        AlgorithmKMeans.__CLASS_NAME__ = self.__class__.__name__

        # copy the inputs into class data
        #
        self.input_d = win_input
        self.output_d = win_output
        self.log_d = win_log
        self.n_clusters = n_clusters
        self.initial = init
        self.n_init = n_init
        self.maxiter = maxiter

        # exit gracefully
        #
        return None

    # method: AlgorithmKMeans::initialize
    #
    # arguments:
    #  data: data recorded from display that will be used for training
    #
    # return:
    #  True
    #
    # This method initializes variables for KMeans
    #
    def initialize(self, data):

        # initialize variables
        #
        self.data = data
        self.classes = len(self.data)

        # find mean within each class
        #
        self.means = [d.mean(axis=0) for d in self.data]

        # set up Kmeans model
        #
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             init=self.initial, n_init=self.n_init,
                             max_iter=self.maxiter,random_state=icf.SEED)
        self.X = np.empty((0, 0))
        self.print_params()

        # exit gracefully
        #
        return True

    # method: AlgorithmKMeans::run_algo
    #
    # arguments:
    #  data: data recorded from display
    #
    # return:
    #  True
    #
    # This method runs the initialization and training
    #
    def run_algo(self, data):

        # initialize and train algorithm
        #
        self.initialize(data)
        self.train()

        # exit gracefully
        #
        return True

    # method: AlgorithmKMeans::train
    #
    # arguments:
    #  None
    #
    # return:
    #  True
    #
    # This method trains the model and computes the cluster mean
    #
    def train(self):

        # stack data and train algorithm
        #
        data = np.vstack((self.data))
        self.kmeans.fit(data)

        # compute cluster means
        #
        self.compute_cluster_mean(self.input_d)

        # exit gracefully
        #
        return True

    def print_params(self):
        param = self.kmeans.get_params()
        self.log_d.append("\n"+(FORMAT.format
                           (PARAMETER, VALUE)))

        for k, v in param.items():
            k, v = str(k), str(v)

            self.log_d.append(FORMAT.format(k, v))

    # method: AlgorithmKMeans::compute_cluster_mean
    #
    # arguments:
    #  ax: display where means will be plotted
    #
    # return:
    #  True
    #
    # This method plots the cluster means
    #
    def compute_cluster_mean(self, ax):

        # find cluster mean and plot
        #
        ax.canvas.axes.scatter(self.kmeans.cluster_centers_[:,0],
                               self.kmeans.cluster_centers_[:,1],
                               c='black', s=8)

        # exit gracefully
        #
        return True

    # method: AlgorithmKMeans::predict
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

        # reshape data
        #
        X = np.concatenate(X, axis=0)
        X = np.reshape(X, (-1, 2))

        # set up the limits and map of values for decision map
        #
        res = (ax.canvas.axes.get_xlim()[1] -
               ax.canvas.axes.get_ylim()[0]) / 100
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))

        # predict values for decision surface
        #
        Z = self.kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # exit gracefully
        #
        return xx, yy, Z

    # method: AlgorithmKMeans::prediction_classifier
    #
    # arguments:
    #  data: data from the evaluation display
    #
    # return:
    #  prediction: the predicated class label
    #
    # This method predicts the class label through calculating the shortest
    # distance between the predicted cluster value and all of the clusters mean
    #
    def prediction_classifier(self, data):

        # find cluster centers
        #
        centers = self.kmeans.cluster_centers_

        # predict cluster label
        #
        data = np.vstack((data))
        clusters = self.kmeans.predict(data)

        # find the class closest to cluster label using euclidean distance
        #
        prediction = []
        for c in clusters:
            predicted_center = centers[c]
            min_dist = np.inf
            for classes in range(len(self.means)):

                # calc distance between mean and predicted value center
                #
                dist = distance.euclidean(self.means[classes], predicted_center)

                # check if calculated distance is lower than current min
                # distance
                #
                if min_dist > dist:
                    min_dist = dist
                    label = classes

            # record prediction
            #
            prediction.append(label)

        # exit gracefully
        #
        return prediction

#
# end of class

#
# end of file
