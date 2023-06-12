#!/usr/bin/env python
#
# file: imld/alg/imld_model.py
#
# revision history:
#
# 20220129 (MM): clean up, implement prepare_data()
# 20200101 (MM): initial version
# This class contains a collection of functions that deal with applying
# algorithms, plotting decision surfaces and step functions used to forward the
# progress of the model
#
#------------------------------------------------------------------------------
#
# imports are listed here
#
#------------------------------------------------------------------------------

# import modules
#
import numpy as np
import gui.imld_gui_window as igw
from PyQt5 import QtWidgets, QtCore

from sklearn.preprocessing import Normalizer
from copy import deepcopy

# ------------------------------------------------------------------------------
#
# global variables are listed here
#
# ------------------------------------------------------------------------------

# step index used in model
#
STEP_SIZE = 1
INITIALIZE = 0
PREPARE_AND_TRAIN = 1
PLOT_TRAIN = 2
COMPUTE_TRAIN_ERROR = 3
COMPUTE_EVAL_ERROR = 4
CONTROL = np.inf

# define model's messages
#
ERROR_RATE_MSG = "\n{} Error Rate = {} / {} = {:.2f}%"
TRAIN = "Training"
EVAL = "Evaluation"
RESET = "\nProcess Resetting..."
BREAK = "==============================================="
EVAL_ERROR = "\nThere's no Eval Data to classify."
EVAL_TRAIN_ERROR = "Eval and Training data classes do not match"

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

#  class: Model
#
#  This class contains methods to apply chosen algorithms, extract the data
#  needed for the algorithms, step through the model, plot decision surfaces and
#  display computed errors.
#
class Model:

    # method: Model::constructor
    #
    # arguments:
    #  algo: the algorithm chosen
    #  win_input: GUI input display
    #  win_output: GUI output display
    #  win_log: GUI process log
    #
    # return: None
    #
    def __init__(self, algo, win_input, win_output, win_log, normalize=False):
        # copy the inputs into class data
        #
        self.input_d = win_input
        self.output_d = win_output
        self.log_d = win_log.output
        self.pbar = win_log.pbar
        self.algo = algo
        self.step_index = 0
        self.info = {TRAIN: self.input_d.class_info,
                     EVAL: self.output_d.class_info}
        QtWidgets.QApplication.processEvents()

        self.normalize = normalize

        # exit gracefully
        #
        return None

    # method: Model::step
    #
    # arguments:
    #  None
    #
    # return:
    #  None
    #
    # This method goes through each step of the algorithmic process
    def step(self):

        # clearing any past results and train the model
        #
        if self.step_index == PREPARE_AND_TRAIN:

            # clear any results in either the input or output display
            #
            if self.input_d.canvas.axes.collections is not None or \
                    self.output_d.canvas.axes.collections is not None:
                self.input_d.clear_result_plot()
                self.output_d.clear_result_plot()

            # prepare the data for the model
            #
            self.progress_bar_update(self.pbar, self.pbar.value(),5)
            QtCore.QCoreApplication.processEvents()
            self.train_data = self.prepare_data(self.info[TRAIN])
            self.progress_bar_update(self.pbar,self.pbar.value(), 5)
            QtCore.QCoreApplication.processEvents()
            self.eval_data = self.prepare_data(self.info[EVAL])

            # check if classes in data match
            #
            if len(self.eval_data) != 0:

                # if classes do not match reset process
                #
                if len(self.train_data) != len(self.eval_data):
                    self.log_d.append(EVAL_TRAIN_ERROR)
                    self.step_index = CONTROL

                    # exit gracefully
                    #
                    return None

            # train the model
            #
            self.progress_bar_update(self.pbar,self.pbar.value(), 10)
            QtCore.QCoreApplication.processEvents()
            self.train()

        # plot the decision surface on the input display using the training data
        #
        elif self.step_index == PLOT_TRAIN:

            self.plot_decision_surface(self.input_d, self.train_data)

        # compute and display the training errors on the process log
        #
        elif self.step_index == COMPUTE_TRAIN_ERROR:
            self.compute_errors(self.train_data, TRAIN) # take in data
            self.progress_bar_update(self.pbar,self.pbar.value(), 5)
            QtCore.QCoreApplication.processEvents()
            self.display_errors(TRAIN)

        # compute and display the evaluation errors and plot the decision
        # surface using the output display and the evaluation data
        #
        elif self.step_index == COMPUTE_EVAL_ERROR:

            # check if the evaluation data is valid
            #
            if len(self.eval_data) > 0:
                self.compute_errors(self.eval_data, EVAL)
                self.progress_bar_update(self.pbar,self.pbar.value(), 5)
                self.display_errors(EVAL)
                self.plot_decision_surface(self.output_d, self.eval_data)
            else:
                self.log_d.append("No evaluation data given")
                QtCore.QCoreApplication.processEvents()
                self.progress_bar_update(self.pbar,self.pbar.value(), 40)


        # pass if step is not listed
        #
        else:
            pass

        # exit gracefully
        #
        return None

    # method: Model::is_done
    #
    # arguments:
    #  None
    #
    # return:
    #  finished: boolean determining if the model finished running
    #
    # This method checks whether the model has finished
    def is_done(self):

        # check if the step index is past computing eval error
        #
        if self.step_index > COMPUTE_EVAL_ERROR:
            finished = True
        else:
            finished = False

        # exit gracefully
        #
        return finished

    # method: Model::increment_step
    #
    # arguments:
    #  None
    #
    # return:
    #  True
    #
    # This method sets the model to the next step of the algorithm
    def increment_step(self):

        # increment step index
        #
        self.step_index += STEP_SIZE

        # exit gracefully
        #
        return True

    # method: Model::reset_step
    #
    # arguments:
    #  None
    #
    # return:
    #  True
    #
    # This method resets the model's process
    def reset_step(self):

        # print out reset message and reset step index
        #
        self.log_d.append(RESET)
        self.log_d.append(BREAK)
        self.step_index = INITIALIZE

        # exit gracefully
        #
        return True

    # method: Model::prepare_data
    #
    # arguments:
    #  info: dict of use data
    #
    # return:
    #  data: list of coordinates from chosen data
    #
    # This method runs the algorithm
    def prepare_data(self, info):

        # set up list of data
        #
        data = []

        # extract the data from the dictionary per class
        #
        for classes in info:
            x_data = np.array(info[classes][1])
            y_data = np.array(info[classes][2])
            coordinates = np.column_stack((x_data, y_data))

            data.append(coordinates)

        # check parse for empty classes
        #
        new_data = [y for y in data if 0 != y.size]

        # normalize data
        #
        if self.normalize:
            # set up parameter for normalizer
            #
            norm = 'l2'

            norm_data = []
            for i in range(len(new_data)):
                norm_data.append(self.normalizer(new_data[i], norm))

            new_data = norm_data

        # exit gracefully
        #
        return new_data

    # method: Model::normalizer
    #
    def normalizer(self, data, norm):
        transformer = Normalizer(norm).fit(data)
        norm_data = transformer.transform(data)

        return norm_data

    # method: Model::train
    #
    # arguments:
    #  None
    #
    # return:
    # True
    #
    # This method runs the algorithm
    def train(self):

        # check if there is training data
        #
        if self.train_data is None:
            return False

        # run the algo with selected data
        #
        self.algo.run_algo(self.train_data)

        # exit gracefully
        #
        return True

    # method: Model::plot_decision_surface
    #
    # arguments:
    #  None
    #
    # return:
    #  True
    #
    # This method plots the decision surface based on the algorithms's
    # prediction
    def plot_decision_surface(self, display, data):

        # record the algorithm's prediction and plot the decision surface
        #
        QtCore.QCoreApplication.processEvents()
        self.progress_bar_update(self.pbar,self.pbar.value(), 10)
        xx, yy, Z = self.algo.predict(display, data)
        QtCore.QCoreApplication.processEvents()
        self.progress_bar_update(self.pbar,self.pbar.value(), 10)
        self.decision_surface(display, xx, yy, Z)

        # exit gracefully
        #
        return True

    # method: Model::compute_errors
    #
    # arguments:
    #  data: data used to calculate errors
    #
    # return: True
    #
    # This method computes the errors of the algorithm
    def compute_errors(self, data, label):

        # verify the number of classes used
        #
        classes = len(data)

        # copy data to get shape of the expected values
        #
        expected = data.copy()

        # fill out class labels
        #
        for i in range(classes):
            expected[i] = len(data[i]) * [i]

        # convert to 1D array
        #
        expected = np.asarray(expected, dtype=object)
        expected = np.hstack(expected)

        # calculate predicted value
        #
        QtCore.QCoreApplication.processEvents()
        prediction = self.algo.prediction_classifier(data)
        QtCore.QCoreApplication.processEvents()
        for i in range(10):
            self.progress_bar_update(self.pbar, self.pbar.value(), 1)

        # calculate total errors
        #
        self.total_error = 0
        batches = 5
        signal = len(expected)// batches

        QtCore.QCoreApplication.processEvents()
        for i in range(len(expected)):
            QtCore.QCoreApplication.processEvents()
            if expected[i] != prediction[i]:
                self.total_error += 1
            QtCore.QCoreApplication.processEvents()
            if i % signal == 0:
                QtCore.QCoreApplication.processEvents()
                self.progress_bar_update(self.pbar, self.pbar.value(), 1)

        self.samples = len(expected)
        self.error = (self.total_error / self.samples) * 100

        # exit gracefully
        #
        return None

    # method: Model::display_errors
    #
    # arguments:
    #  label: either training for evaluation label
    #
    # return: none
    #
    # This method displays the errors calculated
    def display_errors(self, label): # display classification

        # display the error rate for selected label
        #
        text = ERROR_RATE_MSG.format(label, self.total_error, self.samples,
                                     self.error)
        self.log_d.append(text)

        # exit gracefully
        #
        return True

    # method: Model::decision_surface
    #
    # arguments:
    #  ax: the axes that the decision surface is graphed upon
    #  xx: the x coordinate data
    #  yy: the y coordinate data
    #  Z: the height values of the contour
    #
    # return: none
    #
    # This method computes the errors of the algorithm
    def decision_surface(self, ax, xx, yy, Z):

        # reshape the contour
        #
        Z = Z.reshape(xx.shape)

        # plot the decision surface
        #
        ax.canvas.axes.contourf(xx, yy, Z, alpha = 0.4,
                                cmap=self.input_d.surface_color)
        ax.canvas.draw_idle()

        # exit gracefully
        #
        return True

    # method: Model::progress_bar_update
    #
    # arguments:
    #  pbar: the progress bar object
    #  start: the current progress value
    #  finish: target progress value
    #
    # return:
    #  True
    #
    # This method goes through each step of the algorithmic process
    def progress_bar_update(self, pbar, start, finish):

        # increment progress value until target is hit
        #
        for i in range(start, start + finish+1):
            QtCore.QCoreApplication.processEvents()
            pbar.setValue(i)

        return True

#
# end of class

#
# end of file
