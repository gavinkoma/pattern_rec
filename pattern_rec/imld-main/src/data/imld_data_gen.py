#!/usr/bin/env python
#
# file: imld/src/data/imld_data_gen.py
#
# This script defines various preset patterns such as Two Gaussian, Two Ellipses,
# and Toroidal
#
# ------------------------------------------------------------------------------
#
# imports are listed here
#
# ------------------------------------------------------------------------------

# import system modules
#
import numpy as np
import math
import lib.imld_constants_datagen as icd

# ------------------------------------------------------------------------------
#
# global variables are listed here
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
#
# classes are listed here
#
# ------------------------------------------------------------------------------

# class: DataGenerator
#
# This class contains methods to generate the data for several types of
# distributions that can be displayed in the Train and Eval windows
# using the Demo menu.
#
class DataGenerator:

    # method: DataGenerator::constructor
    #
    # arguments:
    #  window_display: window information
    #
    # return: None
    #
    def __init__(self, window_display):

        # store reference to parent loop
        #
        self.display = window_display

    #
    # end of method

    # method: DataGenerator::set_two_gaussian
    #
    # arguments:
    #  npts: number of points in distribution
    #  mean: mean matrix
    #  cov: covariance matrix
    #
    # return: None
    #
    # This method creates and stores data of two normal gaussian distributions.
    #
    def set_two_gaussian(self, npts=None, mean=None, cov=None):

        # makes sure that the size valid
        #
        if npts is np.nan:
            # set to default number of points
            #
            npts = icd.DEFAULT_NPTS_PER_CLASS

        # create two with dim (1, npts) and fill
        #
        class_0_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[0]))
        class_1_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[1]))


        # generate the bi-normal gaussian random deviates for class 0
        #
        if (True in np.isnan(mean)[0]) or (True in np.isnan(cov)[0]):
            mean[0] = icd.DEFAULT_TWOGAUSSIAN_MEAN[0]
            cov[0] = icd.DEFAULT_TWOGAUSSIAN_COV[0]

        data = np.random.multivariate_normal(mean[0], cov[0], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_0 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                                   s=1, gid=class_0_gid,
                                                   color=self.display.class_info[icd.DEFAULT_CLASSES[0]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[0]] = [class_0, data[:, 0], data[:, 1],
                                                           True, self.display.class_info[icd.DEFAULT_CLASSES[0]][4]]

        # generate the bi-normal gaussian random deviates for class 1
        #
        if (True in np.isnan(mean)[1]) or (True in np.isnan(cov)[1]):
            mean[1] = icd.DEFAULT_TWOGAUSSIAN_MEAN[1]
            cov[1] = icd.DEFAULT_TWOGAUSSIAN_COV[1]

        data = np.random.multivariate_normal(mean[1], cov[1], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_1 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                                   s=1, gid=class_1_gid, 
                                                   color=self.display.class_info[icd.DEFAULT_CLASSES[1]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[1]] = [class_1, data[:, 0], data[:, 1],
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[1]][4]]

        # draw on canvas and wait for next command
        #
        self.display.canvas.draw_idle()

    #
    # end of method

    # method: DataGenerator::set_four_gaussian
    #
    # arguments:
    #  npts: number of points in distribution
    #  mean: mean matrix
    #  cov: covariance matrix
    #
    # return: None
    #
    # This method creates and stores data of four normal gaussian
    # distributions.
    #
    def set_four_gaussian(self, npts=None, mean=None, cov=None):

        # makes sure that the size valid
        #
        if npts is np.nan:
            # default to 10000
            #
            npts = icd.DEFAULT_NPTS_PER_CLASS

        # create four arrays with dim (1, npts) and fill
        #
        class_zero_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[0]))
        class_one_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[1]))
        class_two_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[2]))
        class_three_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[3]))

        # generate the bi-normal gaussian random deviates for class 0
        #
        if (True in np.isnan(mean)[0]) or (True in np.isnan(cov)[0]):
            mean[0] = icd.DEFAULT_FOURGAUSSIAN_MEAN[0]
            cov[0] = icd.DEFAULT_FOURGAUSSIAN_COV[0]

        data = np.random.multivariate_normal(mean[0], cov[0], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_0 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_zero_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[0]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[0]] = [class_0, data[:, 0], data[:, 1],
                                             True, self.display.class_info[icd.DEFAULT_CLASSES[0]][4]]

        # generate the bi-normal gaussian random deviates for class 1
        #
        if (True in np.isnan(mean)[1]) or (True in np.isnan(cov)[1]):
            mean[1] = icd.DEFAULT_FOURGAUSSIAN_MEAN[1]
            cov[1] = icd.DEFAULT_FOURGAUSSIAN_COV[1]

        data = np.random.multivariate_normal(mean[1], cov[1], npts)
        #
        class_1 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1], \
                                        s=1, gid=class_one_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[1]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[1]] = [class_1, data[:, 0], data[:, 1], \
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[1]][4]]

        # generate the bi-normal gaussian random deviates for class 2
        #
        if (True in np.isnan(mean)[2]) or (True in np.isnan(cov)[2]):
            mean[2] = icd.DEFAULT_FOURGAUSSIAN_MEAN[2]
            cov[2] = icd.DEFAULT_FOURGAUSSIAN_COV[2]

        data = np.random.multivariate_normal(mean[2], cov[2], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_2 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_two_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[2]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[2]] = [class_2, data[:, 0], data[:, 1],
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[2]][4]]

        # generate the bi-normal gaussian random deviates for class 3
        #
        if (True in np.isnan(mean)[3]) or (True in np.isnan(cov)[3]):
            mean[3] = icd.DEFAULT_FOURGAUSSIAN_MEAN[3]
            cov[3] = icd.DEFAULT_FOURGAUSSIAN_COV[3]

        data = np.random.multivariate_normal(mean[3], cov[3], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_3 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_three_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[3]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[3]] = [class_3, data[:, 0], data[:, 1],
                                              True, self.display.class_info[icd.DEFAULT_CLASSES[3]][4]]

        # draw on canvas and wait for next command
        #
        self.display.canvas.draw_idle()

    #
    # end of method

    # method: DataGenerator::set_over_gaussian
    #
    # arguments:
    #  npts: number of points in distribution
    #  mean: mean matrix
    #  cov: covariance matrix
    #
    # return: None
    #
    # This method creates and stores data of two overlapping
    # gaussian distributions.
    #
    def set_ovlp_gaussian(self, npts=None, mean=None, cov=None):

        # makes sure that the size valid
        #
        if npts is np.nan:

            # set default number of points to 10000
            #
            npts = icd.DEFAULT_NPTS_PER_CLASS

        # create two arrays with dim (1, npts) and fill
        #
        class_zero_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[0]))
        class_one_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[1]))

        # generate the gaussian random deviates for class 0
        #
        if (True in np.isnan(mean)[0]) or (True in np.isnan(cov)[0]):
            mean[0] = icd.DEFAULT_OVERLAPGAUSSIAN_MEAN[0]
            cov[0] = icd.DEFAULT_OVERLAPGAUSSIAN_COV[0]

        data = np.random.multivariate_normal(mean[0], cov[0], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_0 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_zero_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[0]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[0]] = [class_0, data[:, 0], data[:, 1],
                                             True, self.display.class_info[icd.DEFAULT_CLASSES[0]][4]]

        # generate the gaussian random deviates for class 1
        #
        if True not in np.isnan(mean)[1] or True not in np.isnan(cov)[1]:
            mean[1] = icd.DEFAULT_OVERLAPGAUSSIAN_MEAN[1]
            cov[1] = icd.DEFAULT_OVERLAPGAUSSIAN_COV[1]

        data = np.random.multivariate_normal(mean[1], cov[1], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_1 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_one_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[1]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[1]] = [class_1, data[:, 0], data[:, 1],
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[1]][4]]

        # draw on canvas and wait for next command
        #
        self.display.canvas.draw_idle()

    #
    # end of method

    # method: DataGenerator::set_two_ellipses
    #
    # arguments:
    #  npts: number of points in distribution
    #  mean: mean matrix
    #  cov: covariance matrix
    #
    # return: None
    #
    # This method creates and stores data of two elliptical
    # gaussian distributions.
    #
    def set_two_ellipses(self, npts=None, mean=None, cov=None):
        # makes sure that the size valid
        #
        if npts is np.nan:
            # default to 10000
            #
            npts = icd.DEFAULT_NPTS_PER_CLASS

        # create two with dim (1, npts) and fill
        #
        class_zero_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[0]))
        class_one_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[1]))

        # generate the bi-normal gaussian random deviates for class 0
        #
        if (True in np.isnan(mean)[0]) or (True not in np.isnan(cov)[0]):
            mean[0] = icd.DEFAULT_TWOELLIPSE_MEAN[0]
            cov[0] = icd.DEFAULT_TWOELLIPSE_COV[0]

        data = np.random.multivariate_normal(mean[0], cov[0], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_0 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_zero_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[0]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[0]] = [class_0, data[:, 0], data[:, 1],
                                             True, self.display.class_info[icd.DEFAULT_CLASSES[0]][4]]

        # generate the bi-normal gaussian random deviates for class 1
        #
        if (True in np.isnan(mean)[1]) or (True in np.isnan(cov)[1]):
            mean[1] = icd.DEFAULT_TWOELLIPSE_MEAN[1]
            cov[1] = icd.DEFAULT_TWOELLIPSE_COV_COV[1]

        data = np.random.multivariate_normal(mean[1], cov[1], npts)
        # create matplot metadata that holds scatter plot information
        #
        class_1 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_one_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[1]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[1]] = [class_1, data[:, 0], data[:, 1],
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[1]][4]]

        # draw on canvas and wait for next command
        #
        self.display.canvas.draw_idle()

    # method: DataGenerator::set_four_ellipses
    #
    # arguments:
    #  npts: number of points in distribution
    #  mean: mean matrix
    #  cov: covariance matrix
    #
    # return: None
    #
    # This method creates and stores data of four elliptical gaussian
    # gaussian distributions.
    #
    def set_four_ellipses(self, npts=None, mean=None, cov=None):

        # makes sure that the size valid
        #
        if npts is np.nan:
            # set default number of points to 10000
            #
            npts = icd.DEFAULT_NPTS_PER_CLASS

        # create four arrays with dim (1, npts) and fill
        #
        class_zero_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[0]))
        class_one_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[1]))
        class_two_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[2]))
        class_three_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[3]))

        # generate the bi-normal gaussian random deviates for class 0
        #
        if (True in np.isnan(mean)[0]) or (True not in np.isnan(cov)[0]):
            mean[0] = icd.DEFAULT_FOURELLIPSE_MEAN[0]
            cov[0] = icd.DEFAULT_FOURELLIPSE_COV[0]

        data = np.random.multivariate_normal(mean[0], cov[0], npts)
        # create matplot metadata that holds scatter plot information
        #
        class_0 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1], \
                                        s=1, gid=class_zero_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[0]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[0]] = [class_0, data[:, 0], data[:, 1],
                                             True, self.display.class_info[icd.DEFAULT_CLASSES[0]][4]]

        # generate the bi-normal gaussian random deviates for class 1
        #
        if (True in np.isnan(mean)[1]) or (True in np.isnan(cov)[1]):
            mean[1] = icd.DEFAULT_FOURELLIPSE_MEAN[1]
            cov[1] = icd.DEFAULT_FOURELLIPSE_COV[1]

        data = np.random.multivariate_normal(mean[1], cov[1], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_1 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_one_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[1]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[1]] = [class_1, data[:, 0], data[:, 1],
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[1]][4]]

        # generate the bi-normal gaussian random deviates for class 2
        #
        if (True in np.isnan(mean)[2]) or (True not in np.isnan(cov)[2]):
            mean[2] = icd.DEFAULT_FOURELLIPSE_MEAN[2]
            cov[2] = icd.DEFAULT_FOURELLIPSE_COV[2]

        data = np.random.multivariate_normal(mean[2], cov[2], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_2 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_two_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[2]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[2]] = [class_2, data[:, 0], data[:, 1],
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[2]][4]]

        # generate the bi-normal gaussian random deviates for class 3
        #
        if (True in np.isnan(mean)[3]) or (True in np.isnan(cov)[3]):
            mean[3] = icd.DEFAULT_FOURELLIPSE_MEAN[3]
            cov[3] = icd.DEFAULT_FOURELLIPSE_COV[3]

        data = np.random.multivariate_normal(mean[3], cov[3], npts)
        # create matplot metadata that holds scatter plot information
        #
        class_3 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_three_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[3]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[3]] = [class_3, data[:, 0], data[:, 1], True,
                                              self.display.class_info[icd.DEFAULT_CLASSES[3]][4]]

        # draw on canvas and wait for next command
        #
        self.display.canvas.draw_idle()

    #
    # end of method

    # method: DataGenerator::set_rotated_ellipse
    #
    # arguments:
    #  npts: number of points in distribution
    #  mean: mean matrix
    #  cov: covariance matrix
    #
    # return: None
    #
    # This method creates and stores data of one horizontal and
    # one vertical elliptical gaussian distributions.
    #
    def set_rotated_ellipse(self, npts=None, mean=None, cov=None):

        # makes sure that the size valid
        #
        if npts is np.nan:
            # set default number of points to 10000
            #
            npts = icd.DEFAULT_NPTS_PER_CLASS

        # create two arrays with dim (1, npts) and fill
        #
        class_zero_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[0]))
        class_one_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[1]))

        # generate the bi-normal gaussian random deviates
        #
        if (True in np.isnan(mean)[0]) or (True in np.isnan(cov)[0]):
            mean[0] = icd.DEFAULT_ROTATEDELLIPSE_MEAN[0]
            cov[0] = icd.DEFAULT_ROTATEDELLIPSE_COV[0]

        data = np.random.multivariate_normal(mean[0], cov[0], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_0 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_zero_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[0]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[0]] = [class_0, data[:, 0], data[:, 1],
                                             True, self.display.class_info[icd.DEFAULT_CLASSES[0]][4]]

        # generate the bi-normal gaussian random deviates for class 1
        #
        if (True in np.isnan(mean)[1]) or (True in np.isnan(cov)[1]):
            mean[1] = icd.DEFAULT_ROTATEDELLIPSE_MEAN[1]
            cov[1] = icd.DEFAULT_ROTATEDELLIPSE_COV[1]

        data = np.random.multivariate_normal(mean[1], cov[1], npts)

        # create matplot metadata that holds scatter plot information
        #
        class_1 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_one_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[1]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[1]] = [class_1, data[:, 0], data[:, 1],
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[1]][4]]

        self.display.canvas.draw_idle()

    #
    # end of method

    # method: DataGenerator::set_toroidal
    #
    # arguments:
    #  params: array containing number of points for a inner normal
    #          gaussian distribution and an outer ring
    #  mean: mean matrix
    #  cov: covariance matrix
    #
    # return: None
    #
    # This method creates and stores data of one inner ring and
    # one outer ring
    #
    def set_toroidal(self, params, mean, cov):

        # initialize local variables
        #
        x_min = self.display.canvas.axes.get_xlim()[0]
        x_max = self.display.canvas.axes.get_xlim()[1]
        y_min = self.display.canvas.axes.get_ylim()[0]
        y_max = self.display.canvas.axes.get_ylim()[1]

        # checks if user has specified parameters, otherwise defaults
        #
        if np.isnan(params[0]):

            # default total points to 10000
            #
            npts = icd.DEFAULT_NPTS_PER_CLASS
        else:
            npts = int(params[0])

        if np.isnan(params[1]):

            # default number of points in ring to 2000
            #
            num_ring = icd.DEFAULT_TOROID_NPTS_RING
        else:
            num_ring = int(params[1])

        if (np.isnan(params[2]) or np.isnan(params[3])):

            # set default for the radius of the ring
            #
            ring_radius = (icd.DEFAULT_TOROID_OUTER_RADIUS + icd.DEFAULT_TOROID_INNER_RADIUS)/2

            # set default for the standard deviation of the ring
            #
            stddev_radius = (icd.DEFAULT_TOROID_OUTER_RADIUS - icd.DEFAULT_TOROID_INNER_RADIUS) / 2
        else:
            ring_radius = (float(params[2]) + float(params[3]))/2
            stddev_radius = (float(params[3]) - float(params[2]))/2

        # create two arrays with dim (1, npts) and fill
        #
        class_zero_gid = np.full((1, num_ring), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[0]))
        class_one_gid = np.full((1, npts), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[1]))

        # create an empty array
        #
        toroidal = np.empty((num_ring, icd.DEFAULT_TOROID_NUM_CLASSES))

        # create the outside ring
        #
        for i in range(1, num_ring + 1):
            # the angle is linearly distributed from 0 to 2Pi
            #
            angle = np.random.random() * 2 * math.pi
            radius = np.random.normal(ring_radius, stddev_radius)

            # set points
            #
            x_1 = radius * math.cos(angle) + mean[0]
            y_1 = radius * math.sin(angle) + mean[1]

            # plot the points
            #
            toroidal[i - 1, 0] = x_1 + x_min + (x_max - x_min) / 2
            toroidal[i - 1, 1] = y_1 + y_min + (y_max - y_min) / 2

        # create matplot metadata that holds scatter plot information
        #
        class_0 = self.display.canvas.axes.scatter(toroidal[:, 0], toroidal[:, 1],
                                        s=1, gid=class_zero_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[0]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[0]] = [class_0, toroidal[:, 0], toroidal[:, 1],
                                             True, self.display.class_info[icd.DEFAULT_CLASSES[0]][4]]

        # generate the bi-normal gaussian random deviates for class 1
        #
        if True not in np.isnan(mean) or True not in np.isnan(cov):
            data = np.random.multivariate_normal(mean.transpose()[0],
                                                       cov, npts)

        # if there are missing parameters use default values for class 1
        #
        else:
            data = np.random.multivariate_normal(icd.DEFAULT_TOROID_INNER_MEAN,
                                                       icd.DEFAULT_TOROID_INNER_COV,npts)

        # create matplot metadata that holds scatter plot information
        #
        class_1 = self.display.canvas.axes.scatter(data[:, 0], data[:, 1],
                                        s=1, gid=class_one_gid, color= self.display.class_info[icd.DEFAULT_CLASSES[1]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[1]] = [class_1, data[:, 0], data[:, 1],
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[1]][4]]

        # draw on canvas and wait for next command
        #
        self.display.canvas.draw_idle()

    #
    # end of method

    # method: DataGenerator::set_yin_yang
    #
    # arguments:
    #  parameters as retrieved from parameter pop-up
    #
    # returns: None
    #
    # This method creates the data for the Yin-Yang pattern using
    #  user-specified inputs or default values
    #
    def set_yin_yang(self, params):

        # initialize local variables
        #
        x_min = self.display.canvas.axes.get_xlim()[0]
        x_max = self.display.canvas.axes.get_xlim()[1]
        y_min = self.display.canvas.axes.get_ylim()[0]
        y_max = self.display.canvas.axes.get_ylim()[1]

        # the boundary, mean and standard deviation of plot
        #
        xmean = x_min + 0.5 * (x_max - x_min)
        ymean = y_min + 0.5 * (y_max - y_min)
        stddev_center = 1.5 * (x_max - x_min) / 2

        # creating empty lists to save coordinates of points
        #
        yin = []
        yang = []

        # calculate the radius of each class on the plot
        #
        radius1 = 1.5 * ((x_max - x_min) / 4)
        radius2 = .75 * ((x_max - x_min) / 4)

        # define number of samples in each class by checking for user-specified
        #  values and setting defaults if there are none
        #
        if np.isnan(params[0]):
            n_yin = icd.DEFAULT_YING_NPTS
        else:
            n_yin = int(params[0])

        if np.isnan(params[1]):
            n_yang = icd.DEFAULT_YANG_NPTS
        else:
            n_yang = int(params[1])

        class_zero_gid = np.full((1, icd.DEFAULT_NPTS_PER_CLASS), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[0]))
        class_one_gid = np.full((1, icd.DEFAULT_NPTS_PER_CLASS), icd.DEFAULT_CLASSES.index(icd.DEFAULT_CLASSES[1]))

        # producing some random numbers based on Normal distribution and then
        # calculating the points distance to each class, choosing the closest
        # set.
        # the look will exit when both classes has been built up.
        #
        n_yin_counter = 0
        n_yang_counter = 0

        while ((n_yin_counter < n_yin) |
               (n_yang_counter < n_yang)):

            # generate points with Normal distribution
            #
            xpt = np.random.normal(xmean, stddev_center, 1)[0]
            ypt = np.random.normal(ymean, stddev_center, 1)[0]

            # calculate radius for each generated point
            #
            distance1 = np.sqrt(xpt ** 2 + ypt ** 2)
            distance2 = np.sqrt(xpt ** 2 + (ypt + radius2) ** 2)
            distance3 = np.sqrt(xpt ** 2 + (ypt - radius2) ** 2)

            # decide which class each point belongs to
            # when the point added to each class, its counter increases by 1.
            # if the counter reach to the requested number of samples in
            # each class, then no sample will appended to that class and the
            # produced point will be thrown away.
            #
            if distance1 <= radius1:

                if (xpt >= -radius1) & (xpt <= 0):

                    if (((distance1 <= radius1) |
                         (distance2 <= radius2)) &
                            (distance3 > radius2)):

                        if n_yin_counter < n_yin:
                            yin.append([xpt, ypt])
                            n_yin_counter += 1

                    elif n_yang_counter < n_yang:

                        yang.append([xpt, ypt])
                        n_yang_counter += 1

                if (xpt > 0.0) & (xpt <= radius1):

                    if (((distance1 <= radius1) |
                         (distance3 <= radius2)) &
                            (distance2 > radius2)):

                        if n_yang_counter < n_yang:
                            yang.append([xpt, ypt])
                            n_yang_counter += 1

                    elif n_yin_counter < n_yin:
                        yin.append([xpt, ypt])
                        n_yin_counter += 1

        # translate each sample in Yin and Yang from the origin to
        # the center of the plot.
        # for implementing overlap, the overlap parameter meanltiply to one of
        # the plot center points. So the overlap parameter interferes in
        # translation process.
        #
        yang = np.array(yang) + \
               np.array([xmean, ymean])

        yin = np.array(yin) + \
              np.array([xmean, ymean]) * \
              (1 + icd.DEFAULT_YINGYANG_OVERLAP)

        # create matplot metadata that holds scatter plot information
        #
        class_0 = self.display.canvas.axes.scatter(yang[:, 0], yang[:, 1],
                                        s=1, gid=class_zero_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[0]][4])

        class_1 = self.display.canvas.axes.scatter(yin[:, 0], yin[:, 1],
                                        s=1, gid=class_one_gid, color=self.display.class_info[icd.DEFAULT_CLASSES[1]][4])

        # save plot data, x and y coordinates, and color
        #
        self.display.class_info[icd.DEFAULT_CLASSES[0]] = [class_0, yang[:, 0], yang[:, 1],
                                             True, self.display.class_info[icd.DEFAULT_CLASSES[0]][4]]

        self.display.class_info[icd.DEFAULT_CLASSES[1]] = [class_1, yin[:, 0], yin[:, 1],
                                            True, self.display.class_info[icd.DEFAULT_CLASSES[1]][4]]

        # draw on canvas and wait for next command
        #
        self.display.canvas.draw_idle()

    #
    # end of method

#
# end of class

#
# end of file
