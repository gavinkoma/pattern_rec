#!/usr/bin/env python
#
# file: imld/alg/imld_alg_kmeans.py
#
# revision history:
#
# 20220210 (MM): initial version
# This script implements the Class Dependent and Independent Principal
# Component Analysis machine learning algorithm for the ISIP Machine
# Learning Demo software.
#
#------------------------------------------------------------------------------
#
# imports are listed here
#
#------------------------------------------------------------------------------

# import modules
#
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial import distance
from math import cos, sin, atan2, sqrt
import lib.imld_constants_file as icf
import gui.imld_gui_window as igw

#------------------------------------------------------------------------------
#
# global variables are listed here
#
# ------------------------------------------------------------------------------
CI = "CI"
CD = "CD"
NUM_STEPS = 3
PCA_DIM = 2
FORMAT = "{:<15} {:<15}"
PARAMETER = "PARAMETERS"
VALUE = "VALUES"
#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

#  class: AlgorithmPCA
#
#  This class contains methods to apply the PCA algorithm on a set of data
#  while displaying the means and covariances as well as elliptical regions.
#
class AlgorithmPCA():
    # method: AlgorithmPCA::constructor
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
        AlgorithmPCA.__CLASS_NAME__ = self.__class__.__name__

        # copy the inputs into class data
        #
        self.input_d = win_input
        self.output_d = win_output
        self.log_d = win_log
        self.mode = mode

        # exit gracefully
        #
        return None

    # method: AlgorithmPCA::initialize
    #
    # arguments:
    #  data: data recorded from display that will be used for training
    #
    # return:
    #  True
    #
    # initialize variables for PCA
    #
    def initialize(self, data):

        # initialize variables
        #
        self.data = data
        self.cov_mat = None
        self.mean_mat = None
        self.trans_mat = None
        self.dist = None
        self.pca = PCA(PCA_DIM, random_state=icf.SEED)
        self.pca_classes = []
        self.classes = len(self.data)
        self.print_params()

        # exit gracefully
        #
        return True

    # method: AlgorithmPCA::run_algo
    #
    # arguments:
    #  data: data recorded from display
    #
    # return:
    #  True
    #
    # run algorithm steps
    #
    def run_algo(self, data):

        # initialize algorithm
        #
        self.initialize(data)

        # compute and display stats
        #
        self.compute_stats()
        self.display_means()
        self.display_cov()

        # plot ellipse regions
        #
        self.ellipse_regions(data, self.input_d.class_info)

        # exit gracefully
        #
        return True

    # method: AlgorithmPCA::compute_stats
    #
    # arguments: none
    #
    # return:
    #  True
    #
    # this method calculates the covariance and means matrix for each class the PCA is applied to
    #
    def compute_stats(self):

        # initialize variables
        #
        self.mean_mat = []
        self.cov_mat = []
        self.trans_mat = []


        # compute the covariance for each class
        if self.mode == CI:
            data = np.vstack((self.data))
            self.pca.fit(data)
            tmp_cov = self.pca.get_covariance()

        # iterate through class and apply PCA and find the mean,
        for i in range(len(self.data)):
            self.pca.fit(self.data[i])
            self.pca_classes.append(self.pca.fit(self.data[i]))
            mean_vector = self.pca.mean_
            self.mean_mat.append(mean_vector)
            self.trans_mat.append(self.pca.transform(self.data[i]))

            # if CD find the covariance matrix for each class, or if CI append
            # the same covariance matrix
            #
            if self.mode == CI:
                self.cov_mat.append(tmp_cov)

            else:
                self.cov_mat.append(self.pca.get_covariance())

        # exit gracefully
        #
        return True

    # method: AlgorithmPCA::display_means
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

    # method: AlgorithmPCA::display_cov
    #
    # arguments: none
    #
    # return:
    #  True
    #
    # this method displays the covariance matrix of each class in the process log
    #
    def display_cov(self):
        text = 'Covariance Matrix:'

        for classes in range(len(self.cov_mat)):
            text += '\nClass' + str(classes) +': '
            str_cov = np.array2string(self.cov_mat[classes], formatter={'float_kind': lambda x: "%.4f" % x})
            text += '\n' + str_cov

        self.log_d.append(text)

        # exit gracefully
        #
        return True

    def print_params(self):
        param = self.pca.get_params()
        self.log_d.append("\n"+(FORMAT.format
                           (PARAMETER, VALUE)))

        for k, v in param.items():
            k,v = str(k), str(v)

            self.log_d.append(FORMAT.format(k, v))
        return True

    # method: AlgorithmPCA::predict
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
        X = np.concatenate(X, axis=0)
        X = np.reshape(X, (-1, 2))
        res = (ax.canvas.axes.get_xlim()[1] - ax.canvas.axes.get_ylim()[0]) / 100
        x_min, x_max = X[:, 0].min() - .75, X[:, 0].max() + .75
        y_min, y_max = X[:, 1].min() - .75, X[:, 1].max() + .75
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))

        # if dependent reshape the covariance matrix
        #
        if self.mode == CD:
            self.cov_mat = np.reshape(self.cov_mat, (self.classes, 2, 2))
        Z = np.empty((0, 0))

        # calculate the distance using mean of the class and the covariance of the class
        #
        for pairs in np.c_[xx.ravel(), yy.ravel()]:
            distances = np.empty((0, 0))
            for i in range(self.classes):
                maha = (distance.mahalanobis(self.mean_mat[i], pairs, np.linalg.inv(self.cov_mat[i])))
                distances = np.append(distances, maha)

            # save the distance
            #
            Z = np.append(Z, np.argmin(distances))

        # exit gracefully
        #
        return xx, yy, Z

    # method: AlgorithmPCA::prediction_classifier
    #
    # arguments:
    #  data: the class data being used for predictions
    #
    # return:
    #  distance: minimal distance
    #
    # this method calculates the distance prediction of each class
    def prediction_classifier(self, data):

        # set up prediction list
        #
        prediction = []

        # iterate through each sampe
        #
        for i in range(self.classes):
            for j in range(len(data[i])):
                distances = np.empty((0, 0))
                for k in range(self.classes):

                    # calculate the maha distance and record the index of the
                    # minimum value
                    #
                    maha = distance.mahalanobis(self.mean_mat[k],
                                                data[i][j],
                                                np.linalg.inv(self.cov_mat[k]))
                    distances = np.append(distances, maha)

                # record prediction
                #
                prediction.append(np.argmin(distances))


        # exit gracefully
        #
        return prediction

    # method: AlgorithmPCA::ellipse_regions
    #
    # arguments:
    #  x: is the data passed through from the original class data
    #  classes: is the number of classes in the dataset
    #
    # return:
    #  True
    #
    # this method calculates the elliptical region surrounding the class data
    #
    def ellipse_regions(self, x, classes):

        # initialize the boundaries of the support region
        #
        y = np.arange(len(classes))
        x = np.concatenate(x, axis=0)
        x = np.reshape(x, (-1, 2))
        y = np.reshape(y, (-1, 1))

        # initialize the region for each class
        #
        self.support_region = np.empty((0, 0))
        pca_d = [np.empty((2, 2)) * np.nan for i in range(len(classes))]

        # if class independent find the mean and the covariance matrix
        #
        if self.mode == CI:
            val = np.empty((2, 1))
            if x.size > 0:
                mu = np.empty((0, 0))
                for num in range(len(classes)):
                    index = np.where(y == num)[0]
                    reshaped_x = np.reshape(x[index], (-1, 2))

                    # find the mean of the data for the clas
                    #
                    mu_v = np.mean(reshaped_x, axis=0)
                    mu = np.append(mu, mu_v)

                mu = np.reshape(mu, (len(classes), 2, 1))
                self.mu = mu

                # find the covariance matrix
                #
                self.pca.fit_transform(x, y)
                cov = self.pca.get_covariance()
                global_mu = self.pca.mean_

                # find the eigenvalues and eiganvectors
                #
                eigVal, eigVector = np.linalg.eig(cov)

                # create a temporary matrix to hold the initial transformation matrix
                #
                temp = np.empty((2, 2))
                for i in range(2):
                    for j in range(2):
                        temp[j][i] = (eigVector[i][j] / sqrt(eigVal[i]))

                # calculate the theta for the transformation matrix
                #
                alpha = cov[0][0] - cov[1][1]
                beta = -2 * cov[0][1]

                if eigVal[0] > eigVal[1]:
                    theta = atan2((alpha - sqrt((alpha * alpha) +
                                                (beta * beta))), beta)
                else:
                    theta = atan2((alpha + sqrt((alpha * alpha) +
                                                (beta * beta))), beta)

                # calculate the transformation matrix
                #
                trans_matrix = np.linalg.inv(temp)

                self.trans_matrix = temp

            # iterate through each degree
            #
            for i in range(0, 360):
                val[0][0] = 1.5 * cos(i)
                val[1][0] = 1.5 * sin(i)

                # transform the points from the feature space back to the
                # original space to create the support region for the data set
                #
                supp = np.dot(trans_matrix, val)

                # rotate the points (original space)
                #
                xval = (supp[0][0] * cos(theta)) - (supp[1][0] * sin(theta))

                yval = (supp[0][0] * sin(theta)) +(supp[1][0] * cos(theta))

                xval = xval + global_mu[0]
                yval = yval + global_mu[1]

                self.support_region = np.append(self.support_region, [xval, yval])

            self.support_region = np.reshape(self.support_region, (-1, 2))

        # if class dependent
        #
        else:
            val = np.empty((2, 1))
            for i in range(len(classes)):
                index = np.where(y == i)[0]
                if x[index].size > 0:

                    # find the eigenvalues and eigenvectors
                    #
                    self.Eigval = self.pca_classes[i].explained_variance_
                    self.Eigvect = self.pca_classes[i].components_
                    for j in range(2):
                        for k in range(2):
                            pca_d[i][k][j] = self.Eigvect[k][j] / sqrt(self.Eigval[j])

                # find the mean and covariance matrices
                #
                if not np.isnan(pca_d[i])[0][0]:
                    covPCA = self.cov_mat[i]
                    meanPCA = self.mean_mat[i]

                    # calculate the inverse transformation matrix
                    #
                    alpha = covPCA[0][0] - covPCA[1][1]
                    beta = -2 * covPCA[0][1]

                    if self.Eigval[0] > self.Eigval[1]:
                        theta = atan2((alpha - sqrt((alpha * alpha) + (beta * beta))), beta)
                    else:
                        theta = atan2((alpha + sqrt((alpha * alpha) + (beta * beta))), beta)

                    inv_trans = np.linalg.inv(pca_d[i])

                # iterate through each degree for the support region
                #
                for z in range(0, 360):
                    val[0][0] = 1.5 * cos(z)
                    val[1][0] = 1.5 * sin(z)

                    # transform the points from the feature space back to the
                    # original space to create the support region for the data set
                    #
                    supp = np.dot(inv_trans, val)

                    # rotate the points (original space)
                    #
                    xval = (supp[0][0] * cos(theta)) - (supp[1][0] * sin(theta))

                    yval = (supp[0][0] * sin(theta)) + (supp[1][0] * cos(theta))

                    xval = xval + meanPCA[0]
                    yval = yval + meanPCA[1]

                    self.support_region = np.append(self.support_region, [xval, yval])

                self.support_region = np.reshape(self.support_region, (-1, 2))

        # plot the support region
        #
        self.input_d.canvas.axes.scatter(self.support_region[:, 0], self.support_region[:, 1], c='black', s=1)
        self.input_d.canvas.draw_idle()

        # exit gracefully
        #
        return True

#
# end of class

#
# end of file
