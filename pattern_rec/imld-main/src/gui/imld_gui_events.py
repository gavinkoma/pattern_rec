#!/usr/bin/env python
#
# file: event_loop.py
#
# This script implements the event loop of the ISIP Machine Learning Demo
# user interface. The event loop runs until the application is quit.
#-----------------------------------------------------------------------------

# import system modules
#
import numpy as np
import sys
import time as t

from sklearn.preprocessing import Normalizer
from copy import deepcopy

# import GUI modules
#
from PyQt5 import QtGui, QtCore, QtWidgets

# import locally defined modules
#
import gui.imld_gui_window as igw
import gui.imld_gui_params as igp

from data.imld_data_gen import DataGenerator
import data.imld_data_io as idio

from alg.imld_alg_pca import AlgorithmPCA
from alg.imld_alg_svm import AlgorithmSVM
from alg.imld_alg_knn import AlgorithmKNN
from alg.imld_alg_kmeans import AlgorithmKMeans
from alg.imld_alg_lda import AlgorithmLDA
from alg.imld_alg_mlp import AlgorithmMLP
from alg.imld_alg_rf import AlgorithmRF
import alg.imld_model as model


#-----------------------------------------------------------------------------
#
# define global variables
#
#-----------------------------------------------------------------------------
SELECTED_CLASS=""
TOTAL_CLASSES=[]
SELECTED_NAME=""
TOTAL_NAMES=[]
STANDARD_COLORS = ["red","blue","yellow","black","orange","purple","green","magenta" ]

# this text appears in a pop-up window when users select IMLD > About
#
ABOUT_TEXT = "This software is intended to be used for \
educational purposes. Please direct any feedback or questions to \
help@nedcdata.org."

IMLD_VERSION = "v1.8.0"

#-----------------------------------------------------------------------------
#
# classes are listed here
#
#-----------------------------------------------------------------------------

# class: EventLoop
#
# This class contains a variety of methods including those to prompt the user,
#  set up for data to be plotted, and to show data in the Train and Eval
#  windows of the application.
#
class EventHandler(QtCore.QEventLoop):

    # define class variables
    #
    def __init__(self):
        QtCore.QEventLoop.__init__(self)

        # create instance of the main window user interface and DataPoint
        #
        self.ui = igw.MainWindow(self)
        self.data_points_t = DataGenerator(self.ui.input_display)
        self.data_points_e = DataGenerator(self.ui.output_display)
        self.item_now = [None, None]
        self.io = idio.DataIO(self.ui)
        self.load_flag = False
        self.color_bank = []
        self.saved_color = 'color'
        self.standard_colors = ["red","blue","yellow","black","orange","purple","green","magenta" ]

        self.is_normalized = False

        # current algorithm
        #
        self.algo = None
        self.data = False
        self.initialized = False
        self.t_classes = []
        self.classes = None

        # define variables for gaussian generator
        #
        self.mu = np.nan
        self.cov = np.nan
        self.points = np.nan
        self.params = np.nan

    # method: EventLoop::display_warning
    # arguments:
    #  message: a message to display to the user
    #
    # This method sends warning message to user.
    #
    def display_warning(self, message):

        # instantiate QMessageBox                                            
        #
        self.warning = QtWidgets.QMessageBox()

        # set title and sub text of pop-up window                           
        #                                                                    
        self.warning.setText("Warning:")
        self.warning.setInformativeText(message)

        # set window size                                                    
        #                                                                    
        self.warning.setStyleSheet("QLabel{min-width:200 px;"
                                   "min-height:30 px;}")

        # show window                                                          
        #                                                                    
        self.warning.show()

    # method: EventLoop::display_error
    #
    # send error message to user
    #
    def display_error(self, message):

        # instantiate QMessageBox                                           
        #
        self.error = QtWidgets.QMessageBox()

        # set title and sub text of pop-up window
        #                                                                     
        self.error.setText("Error:")
        self.error.setInformativeText(message)

        # set window size                                                   
        #                                                                     
        self.error.setStyleSheet("QLabel{min-width:200 px; min-height:30 px;}")

        # show window                                                          
        #                                                                    
        self.error.show()

    # method: EventLoop::about_IMLD
    #
    # This method creates pop-up window for the IMLD > About menu option.
    #
    def about_IMLD(self):

        # instantiate QMessageBox
        #
        self.dialog = QtWidgets.QMessageBox()

        # set title and sub text of pop-up window
        #
        self.dialog.setText( "ISIP Machine Learning Demonstration")
        self.dialog.setInformativeText("Version: " + IMLD_VERSION +
                                       "\n\n" + ABOUT_TEXT)
        self.dialog.setWindowTitle("About IMLD")

        # set window size
        #
        self.dialog.setStyleSheet("QLabel{min-width:500 px; min-height:60 px;}")

        # remove "Ok" button (needs work)
        #
        button = self.dialog.defaultButton()
        self.dialog.removeButton(button)

        # show window
        #
        self.dialog.show()

    # method: EventLoop::EventLoop::prompt_for_load_train
    #
    # This method contains a pop-up window for users to load data into the  
    # Eval window of the UI and stores the data.
    #
    def prompt_for_load_train(self):

        try:
            # read in the data from the file
            #
            classes, colors, limits, self.train= self.io.load_file('train')
            
            # clear the window
            #

            # if there is data in the window:
            #
            if len(self.ui.input_display.class_info) != 0:
                check_data = False

                for class_name in self.ui.input_display.class_info:
                    if np.size(self.ui.input_display.class_info[class_name][1]) > 0\
                        and np.size(self.ui.input_display.class_info[class_name][2]) > 0:
                        check_data = True
                if check_data:
                    message = "Current data is replaced with the loaded data "
                    self.display_warning(message)

                    self.ui.input_display.clear_plot()
                    self.ui.input_display.class_info = {}

            # get a list of class
            #
            self.train_class = []

            # if classes is not declared in file
            #
            if classes == []:

                # find classes from data/self.train
                #
                self.train_class = list(self.train.keys())
            else:
                self.train_class = classes

                # check if classes from the comment doesnt match with data
                #
                if set(self.train_class) != set(list(self.train.keys())):

                    # print out error
                    #
                    message = "Classes declared in the comment does not match with data, use classes in data"
                    self.display_warning(message)

                    self.train_class = list(self.train.keys())

            # get a dict of color
            #
            train_color = {}

            # if color is not declared in file
            #
            if colors == []:
                for class_i in self.train_class:
                    
                    # check if color is declared in eval window
                    #
                    if class_i in list(self.ui.outtput_display.class_info.keys()):
                        new_color = self.ui.output_display.class_info[class_i][4]
                    else:
                        new_color = self.preset_color()

                    if class_i not in train_color:
                        train_color[class_i] = []

                    train_color[class_i] = new_color
            
            # if color is declared
            #
            else:
                for idx, class_i in enumerate(self.train_class):
                    
                    if class_i not in train_color:
                        train_color[class_i] = []

                    if (idx < len(colors)): 
                        train_color[class_i] = colors[idx]
                    
                    # if color declared in the comment is not enough
                    #
                    else:
                        if class_i in list(self.ui.output_display.class_info.keys()):
                            new_color = self.ui.output_display.class_info[class_i][4]
                        else:
                            new_color = self.preset_color()
                            while(new_color in train_color.values()):
                                new_color = self.preset_color()
                        train_color[class_i] = new_color

            # if limits is not declared, use min and max of each coordinate
            #
            if limits == []:
                limits_for_all_class = []

                # find min/max of x and y for each class
                #
                for class_name in self.train_class:
                    min_x = min(self.train[class_name], key=lambda x:x[0])[0]
                    max_x = max(self.train[class_name], key=lambda x:x[0])[0]

                    min_y = min(self.train[class_name], key=lambda x:x[1])[1]
                    max_y = max(self.train[class_name], key=lambda x:x[1])[1]

                    limits_for_all_class.append([min_x, max_x, min_y, max_y])
                
                # choose the limits based on all the classes
                #
                min_x = min(limits_for_all_class, key=lambda x:x[0])[0]
                max_x = max(limits_for_all_class, key=lambda x:x[1])[1]
                min_y = min(limits_for_all_class, key=lambda x:x[2])[2]
                max_y = max(limits_for_all_class, key=lambda x:x[3])[3]
                limits = [min_x-1, max_x+1, min_y-1, max_y+1]

            # set limits in training window
            #
            self.ui.input_display.x_axis = [float(limits[0]), float(limits[1])]
            self.ui.input_display.y_axis = [float(limits[2]), float(limits[3])]
            self.ui.input_display.initUI()

            # stores important variables (train_c holds the classes)
            #
            if self.train is not None:

                # set up dictionary variables to store training data
                #
                self.current_c = None
                train_x = np.empty((0, 0))
                train_y = np.empty((0, 0))
                self.once_c = False
                self.load_flag = True

                for class_name in self.train_class:

                    train_x = np.empty((0, 0))
                    train_y = np.empty((0, 0))

                    # Add the classes to both the GUI toolbar and the display dictionary
                    #
                    self.retrieve_class_parameters(class_name, train_color[class_name])
                    self.ui.input_display.class_info[class_name] = [self.current_c, train_x, train_y, self.once_c, train_color[class_name]]
                    
                    # get list of x coordinate from self.train at class_name
                    #
                    train_x = [[point[0] for point in self.train[class_name]]]

                    # get list of y coordinate from self.train at class_name
                    #
                    train_y = [[point[1] for point in self.train[class_name]]]

                    # populate the classes in the dictionary with data
                    #
                    # append train x coordinate to class_info
                    #
                    self.ui.input_display.class_info[class_name][1] = np.append\
                        (self.ui.input_display.class_info[class_name][1], train_x)

                    # append train y coordinate to class_info
                    #
                    self.ui.input_display.class_info[class_name][2] = np.append\
                        (self.ui.input_display.class_info[class_name][2], train_y)

                # plot the training data in the Input display
                #
                self.plot_train_data()
                self.load_flag = False
                
        except:
            print("Warning: File is not loaded")

    # method: EventLoop::prompt_for_load_eval
    #
    # This method contains a pop-up window for users to load data into the
    # Eval window of the UI and stores the data.
    #

    def prompt_for_load_eval(self):

        try:
            # read in the data from the file
            #
            classes, colors, limits, self.eval= self.io.load_file('eval')

            # clear the window
            #

            # if there is data in the window
            #
            if len(self.ui.output_display.class_info) != 0:
                check_data = False

                for class_name in self.ui.output_display.class_info:
                    if np.size(self.ui.output_display.class_info[class_name][1]) > 0\
                        and np.size(self.ui.output_display.class_info[class_name][2]) > 0:
                        check_data = True

                if check_data:
                    message = "Current data is replaced with the loaded data "
                    self.display_warning(message)
                
                    self.ui.output_display.clear_plot()
                    self.ui.output_display.class_info = {}

            # get a list of class
            #
            self.eval_class = []

            # if classes is not declared in file
            #
            if classes == []:

                # find classes from data/self.eval
                #
                self.eval_class = list(self.eval.keys())
            else:
                self.eval_class = classes

                # check if classes from the comment doesnt match with data
                #
                if set(self.eval_class) != set(list(self.eval.keys())):

                    # print out error
                    #
                    message = "Classes declared in the comment does not match with data, use classes in data"
                    self.display_warning(message)

                    self.eval_class = list(self.eval.keys())

            # get a dict of color
            #
            eval_color = {}

            # if color is not declared in file
            #
            if colors == []:
                for class_i in self.eval_class:

                    # check if color is declared in train window
                    #
                    if class_i in list(self.ui.input_display.class_info.keys()):
                        new_color = self.ui.input_display.class_info[class_i][4]
                    else:
                        new_color = self.preset_color()

                    if class_i not in eval_color:
                        eval_color[class_i] = []

                    eval_color[class_i] = new_color
            
            # if color is declared
            #
            else:
                for idx, class_i in enumerate(self.eval_class):
                    if class_i not in eval_color:
                        eval_color[class_i] = []
                    
                    if (idx < len(colors)): 
                        eval_color[class_i] = colors[idx]
                    
                    # if color declared in the comment is not enough
                    #
                    else:

                        # check if the color is in the train window
                        #
                        if class_i in list(self.ui.input_display.class_info.keys()):
                            new_color = self.ui.input_display.class_info[class_i][4]
                        else:
                            new_color = self.preset_color()
                            while(new_color in eval_color.values()):
                                new_color = self.preset_color()
                        eval_color[class_i] = new_color

            # if limits is not declared, use min and max of each coordinate
            #
            if limits == []:
                limits_for_all_class = []

                # find min/max of x and y for each class
                #
                for class_name in self.eval_class:
                    min_x = min(self.eval[class_name], key=lambda x:x[0])[0]
                    max_x = max(self.eval[class_name], key=lambda x:x[0])[0]

                    min_y = min(self.eval[class_name], key=lambda x:x[1])[1]
                    max_y = max(self.eval[class_name], key=lambda x:x[1])[1]

                    limits_for_all_class.append([min_x, max_x, min_y, max_y])
                
                # choose the limits based on all the classes
                #
                min_x = min(limits_for_all_class, key=lambda x:x[0])[0]
                max_x = max(limits_for_all_class, key=lambda x:x[1])[1]
                min_y = min(limits_for_all_class, key=lambda x:x[2])[2]
                max_y = max(limits_for_all_class, key=lambda x:x[3])[3]
                limits = [min_x-1, max_x+1, min_y-1, max_y+1]

            # set limits in eval window
            #
            self.ui.output_display.x_axis = [float(limits[0]), float(limits[1])]
            self.ui.output_display.y_axis = [float(limits[2]), float(limits[3])]
            self.ui.output_display.initUI()

            # stores important variables (train_c holds the classes)
            #
            if self.eval is not None:

                # set up dictionary variables to store training data
                #
                self.current_c = None
                eval_x = np.empty((0, 0))
                eval_y = np.empty((0, 0))
                self.once_c = False
                self.load_flag = True

                for class_name in self.eval_class:

                    eval_x = np.empty((0, 0))
                    eval_y = np.empty((0, 0))

                    # Add the classes to both the GUI toolbar and the display dictionary
                    #
                    self.retrieve_class_parameters(class_name, eval_color[class_name])
                    self.ui.output_display.class_info[class_name] = [self.current_c, eval_x, eval_y, self.once_c, eval_color[class_name]]

                    # get list of x coordinate from self.train at class_name
                    #
                    eval_x = [[point[0] for point in self.eval[class_name]]]

                    # get list of y coordinate from self.train at class_name
                    #
                    eval_y = [[point[1] for point in self.eval[class_name]]]

                    # populate the classes in the dictionary with data
                    #
                    # append train x coordinate to class_info
                    #
                    self.ui.output_display.class_info[class_name][1] = np.append\
                        (self.ui.output_display.class_info[class_name][1], eval_x)

                    # append train y coordinate to class_info
                    #
                    self.ui.output_display.class_info[class_name][2] = np.append\
                        (self.ui.output_display.class_info[class_name][2], eval_y)

                # plot the training data in the Input display
                #
                self.plot_eval_data()
                self.load_flag = False
        except:
            print("Warning: File is not loaded")


    # method: EventLoop::prompt_for_save_train
    #
    # This method contains a secondary window that allows the user to save
    # data that is currently being stored in the Train window.
    #
    def prompt_for_save_train(self):

        limits = [self.ui.input_display.x_axis[0], self.ui.input_display.x_axis[-1],
                  self.ui.input_display.y_axis[0], self.ui.input_display.y_axis[-1]]
        self.io.save_file(self.ui.input_display.class_info, 'train', limits)

    # method: EventLoop::prompt_for_save_eval
    #
    # This method contains a secondary window that allows the user to save     
    # data that is currently being stored in the Eval window.
    #
    def prompt_for_save_eval(self):

        limits = [self.ui.output_display.x_axis[0], self.ui.output_display.x_axis[-1],
                  self.ui.output_display.y_axis[0], self.ui.output_display.y_axis[-1]]
        self.io.save_file(self.ui.output_display.class_info, 'eval', limits)

    # method: EventLoop::plot_train_data
    #
    # This method is used to plot data that was loaded by the user  with the
    #  load_train_data method.
    #
    def plot_train_data(self):

        # iterate through the class dictionary and plot the class data
        #
        for classes in self.ui.input_display.class_info:
            self.ui.input_display.class_info[classes][0] = self.ui.input_display.canvas.axes.scatter(None,None, s=1)
            # set the color up for each class
            #
            self.ui.input_display.class_info[classes][0].set_color(self.ui.input_display.class_info[classes][4])
            self.ui.input_display.class_info[classes][0].set_offsets(np.column_stack
                                                                     ((self.ui.input_display.class_info[classes][1],
                                                                       self.ui.input_display.class_info[classes][2]))
                                                                     )
            index = self.train_class.index(classes)
            self.ui.input_display.class_info[classes][0].set_gid\
                (np.full((1, np.shape(self.ui.input_display.class_info[classes][1])[0]),index))

        self.ui.input_display.canvas.draw_idle()

    # method: EventLoop::plot_eval_data
    #
    # This method is used to plot data that was loaded by the user  with the
    # load_eval_data method.
    #
    def plot_eval_data(self):

        # iterate through the class dictionary
        #
        for classes in self.ui.output_display.class_info:

            # set up the color for each class
            #
            self.ui.output_display.class_info[classes][0] = self.ui.output_display.canvas.axes.scatter(None, None, s=1)
            self.ui.output_display.class_info[classes][0].set_color(self.ui.output_display.class_info[classes][4])
            self.ui.output_display.class_info[classes][0].set_offsets(np.column_stack
                                                                     ((self.ui.output_display.class_info[classes][1],
                                                                       self.ui.output_display.class_info[classes][2]))
                                                                )
            index = self.eval_class.index(classes)
            self.ui.output_display.class_info[classes][0].set_gid \
                (np.full((1, np.shape(self.ui.output_display.class_info[classes][1])[0]), index))
        self.ui.output_display.canvas.draw_idle()

    # method: EventLoop::retrieve_class_parameters
    #
    # arguments:
    #  name: the name of the targeted class
    #
    # This method extracts the parameters from the input window that appears
    # when a new class is being added.
    #
    def retrieve_class_parameters(self,name,color=None):

        # call the global variables for class selections
        #
        global SELECTED_CLASS
        global TOTAL_NAMES
        global SELECTED_NAME
        global TOTAL_CLASSES

        # check if the name of the class is not empty, not in used Names and does not have a Color
        #
        if len(name) != 0 and (str(name) not in TOTAL_NAMES) and self.ui.input_display.color_c is not False:
            
            # create the widget for the new class
            #
            classes = QtWidgets.QAction(name, checkable=True)
            classes.setObjectName(name)

            # update the global variables
            #
            TOTAL_NAMES.append(name)
            TOTAL_CLASSES.append(classes)
            SELECTED_CLASS = classes
            SELECTED_NAME = name

            # update the input and output display variables
            #
            self.ui.input_display.t_current_class = TOTAL_NAMES
            self.ui.output_display.t_current_class = TOTAL_NAMES
            self.ui.input_display.current_class = name
            self.ui.output_display.current_class = name
            self.ui.input_display.all_classes = TOTAL_CLASSES
            self.ui.output_display.all_classes = TOTAL_CLASSES
            self.ui.input_display.class_info[name] = None
            self.ui.output_display.class_info[name] = None

            # add the class widget to the menu and trigger it
            #
            self.ui.menu_bar.class_menu.addAction(self.ui.menu_bar.class_group.addAction(classes))
            classes.triggered.connect(self.active_class_data)
            classes.trigger()

            # check if a color is chosen, if not use one from preset
            #
            if self.ui.input_display.color_c in self.color_bank or self.ui.input_display.color_c == None:
                if self.load_flag == False:
                    col = self.preset_color()
                    self.ui.input_display.color_c = self.ui.output_display.color_c = col
                    #message = "Color chosen is invalid, instead using a preset color: %s" % self.ui.input_display.color_c
                    #self.display_warning(message)
                else:
                    if color is not None:
                        self.color_bank.append(color)
                        self.ui.input_display.colors_used.append(color)
                        self.ui.output_display.colors_used.append(color)


            if color is None:
            # set up the class dictionary
            #
                self.ui.input_display.class_info[name] = [self.ui.input_display.current_class, self.ui.input_display.x, self.ui.input_display.y, self.ui.input_display.once_c, self.ui.input_display.color_c]
                self.ui.output_display.class_info[name] = [self.ui.output_display.current_class, self.ui.output_display.x, self.ui.output_display.y, self.ui.output_display.once_c, self.ui.output_display.color_c]
                d_values = list(self.ui.input_display.class_info.values())
                self.color_bank = [i[-1] for i in d_values]
            else:
                self.ui.input_display.class_info[name] = [self.ui.input_display.current_class, self.ui.input_display.x, self.ui.input_display.y, self.ui.input_display.once_c, color]
                self.ui.output_display.class_info[name] = [self.ui.output_display.current_class, self.ui.output_display.x, self.ui.output_display.y, self.ui.output_display.once_c, color]
            self.ui.process_desc.output.append("Classes: added class '%s'" % name)

            # update color bank
            #


    # method:EventLoop::handled_color
    #
    # argumenets:
    #  name: the name of the color
    #
    # This method takes the color input and makes the newly added class that color
    def handled_color(self, name):

        # check if the name is not color
        #
        if name != "color":

            # check if the color has been used
            #
            if name in self.ui.input_display.colors_used :
                self.ui.process_desc.output.append("Choose a color that has not been picked yet")
                self.ui.process_desc.output.append("Colors already picked"+ str(self.ui.input_display.colors_used))
                self.ui.input_display.color_c = False
                self.ui.output_display.color_c = False

            # update variables with the chosen color
            #
            else:
                self.saved_color = name
             #   self.ui.input_display.colors_used.append(name)
            #    self.ui.output_display.colors_used.append(name)
             #   self.ui.input_display.color_c = name
             #   self.ui.output_display.color_c = name

        # choose color if not chosen
        #
        else:
            self.ui.process_desc.output.append("Please choose a color")
            self.ui.input_display.color_c = False
            self.ui.output_display.color_c = False

    # method:EventLoop::preset_color
    #
    # argumenets: none
    #
    # This method sets up the preset color
    def handled_color_saver(self):
        if self.saved_color != "color":
            self.ui.input_display.colors_used.append(self.saved_color)
            self.ui.output_display.colors_used.append(self.saved_color)
            self.ui.input_display.color_c = self.saved_color
            self.ui.output_display.color_c = self.saved_color
            self.saved_color = 'color'

    # method:EventLoop::preset_color
    #
    # argumenets: none
    #
    # This method sets up the preset color

    def preset_color(self):

        # pop color from color stack and check if its in the color bank
        # if so keep going until a valid color is chosen
        #
        color = self.standard_colors.pop()

        if color in self.color_bank:
            col = self.preset_color()
        else:
            self.color_bank.append(color)
            self.ui.input_display.colors_used.append(color)
            self.ui.output_display.colors_used.append(color)
            return color
        return col

    # method:EventLoop::reset_color
    #
    # arguments: none
    #
    # reset the colors chosen
    def reset_color(self):
        self.ui.input_display.color_c = None
        self.ui.output_display.color_c = None

    # method:EventLoop::handled_surface_color
    #
    # arguments: none
    #
    # check surface color
    def handled_surface_color(self, name):
        if name not in igw.colormaps():
            self.ui.input_display.surface_color = 'winter'
        else:
            self.ui.input_display.surface_color = name

    # method: EventLoop::handled_signal
    #
    # arguments:
    #  checked: the specific signal of the button pressed
    #  action: the button that is being checked
    #
    # This method makes the action and records its signal
    #
    def handled_signal(self, checked, action):

        # update current class with the triggered item
        #
        self.item_now[0]= action
        self.item_now[1]= action.objectName()
        self.ui.input_display.current_class = self.item_now[1]
        self.ui.output_display.current_class = self.item_now[1]


    # method: EventLoop::remove_classes
    #
    # arguments:
    #  checked: the signal of the triggered item
    #
    # remove the chosen class
    def remove_classes(self, checked, all=None):

        # remove all info from chosen class
        #
        if self.ui.output_display.class_info and self.ui.input_display.class_info:

            self.ui.menu_bar.class_menu.removeAction(self.item_now[0])
            self.item_now[0].deleteLater()

            self.ui.process_desc.output.append("Classes: deleted class '%s'" % self.item_now[1])

            removed_color = self.ui.input_display.class_info[self.item_now[1]][-1]

            if len(self.standard_colors) != len(STANDARD_COLORS):
                self.standard_colors.append(removed_color)
            self.color_bank.remove(removed_color)
            self.ui.input_display.remove_class(self.item_now[1])
            self.ui.output_display.remove_class(self.item_now[1])

            if all == None:
                self.ui.menu_bar.class_group.actions()[0].trigger()

        else:
            pass

    # method: EventLoop::clear_input
    # arguments:
    #  checked: the specific signal of the button pressed
    # This method clears all present information visible on input plot
    #
    def clear_input(self,checked):
        self.ui.input_display.clear_plot()

    # method: EventLoop::clear_output
    # arguments:
    #  checked: the specific signal of the button pressed
    # This method clears all present information visible on output plot
    #
    def clear_output(self, checked):
        self.ui.output_display.clear_plot()

    # method: EventLoop::clear_input_result
    # arguments:
    #  checked: the specific signal of the button pressed
    #
    # TThis method clears the results visible on input plot
    #
    def clear_input_result(self,checked):
        self.ui.input_display.clear_result_plot()


    # method: EventLoop::clear_output_result
    # arguments:
    #  checked: the specific signal of the button pressed
    #
    # This method clears the results visible on output plot
    #
    def clear_output_result(self, checked):
        self.ui.output_display.clear_result_plot()

    # method: EventLoop::reset_window
    # arguments:
    #  checked: the specific signal of the button pressed
    #
    # This method resets the app externally and internally
    #
    def reset_window(self, checked):

        class_list = self.ui.menu_bar.class_group.actions()
        all = True
        for i in class_list:
            i.triggered.connect(lambda check: self.remove_classes(check,all))
            i.trigger()

        self.ui.input_display.class_info = {}
        self.ui.output_display.class_info = {}
        self.color_bank = []
        self.reset_color()
        self.standard_colors = STANDARD_COLORS.copy()

        self.ui.process_desc.clear_progress()

    # method: EventLoop::active_class_data
    # This method selects the active class when it is either
    # added or selected from the menu
    #
    def active_class_data(self):
        self.classes = True

    # method: EventLoop::plot_two_gaus
    #
    # arguments:
    #  window: the input window selected by the user (train or eval)
    #
    # This method plots the two gaussian preset pattern.
    #
    def plot_two_gaus(self, window):

        # retrieve user submitted parameters
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.menu_bar.clear_train_all.trigger()
        else:
            self.ui.menu_bar.clear_eval_all.trigger()
        self.retrieve_parameters()
        self.retrieve_class_parameters("Class0")
        self.retrieve_class_parameters("Class1")


        # if user selects train option, plot in train window
        #
        if window == igp.WINDOW_TRAIN:
            self.data_points_t.set_two_gaussian(self.points, self.mu, self.cov)

        # otherwise plot in eval window
        #
        else:
            self.data_points_e.set_two_gaussian(self.points, self.mu, self.cov)



    # method: EventLoop::plot_four_gauss
    #
    # arguments:
    #  window: the input window selected by the user (train or eval)
    #
    # This method plots the four gaussian preset pattern.
    #
    def plot_four_gaus(self, window):

        # retrieve user submitted parameters                                  
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.menu_bar.clear_train_all.trigger()
        else:
            self.ui.menu_bar.clear_eval_all.trigger()
        self.retrieve_parameters()
        self.retrieve_class_parameters("Class0")
        self.retrieve_class_parameters("Class1")
        self.retrieve_class_parameters("Class2")
        self.retrieve_class_parameters("Class3")

        # if user selects train option, plot in train window  
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.input_display.clear_plot()
            self.data_points_t.set_four_gaussian(self.points,
                                                 self.mu, self.cov)
        # otherwise plot in eval window
        #
        else:
            self.ui.output_display.clear_plot()
            self.data_points_e.set_four_gaussian(self.points,
                                                 self.mu, self.cov)

    # method: EventLoop::plot_overe_gaus
    #
    # arguments:
    #  window: the input window selected by the user (train or eval)
    #
    # This method plots the over gaussian preset pattern.
    #
    def plot_ovlp_gaussian(self, window):

        # retrieve user submitted parameters                                 
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.menu_bar.clear_train_all.trigger()
        else:
            self.ui.menu_bar.clear_eval_all.trigger()
        self.retrieve_parameters()
        self.retrieve_class_parameters("Class0")
        self.retrieve_class_parameters("Class1")

        # if user selects train option, plot in train window                
        #                                                                    
        if window == igp.WINDOW_TRAIN:
            self.ui.input_display.clear_plot()
            self.data_points_t.set_ovlp_gaussian(self.points, self.mu, self.cov)
        # otherwise plot in eval window                                   
        #                                                                    
        else:
            self.ui.output_display.clear_plot()
            self.data_points_e.set_ovlp_gaussian(self.points,
                                                 self.mu, self.cov)

    # method: EventLoop::plot_two_ellip
    #
    # arguments:
    #  window: the input window selected by the user (train or eval)
    #
    # This method plots the two ellipse preset pattern.
    #
    def plot_two_ellip(self, window):

        # retrieve user submitted parameters                      
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.menu_bar.clear_train_all.trigger()
        else:
            self.ui.menu_bar.clear_eval_all.trigger()
        self.retrieve_parameters()
        self.retrieve_class_parameters("Class0")
        self.retrieve_class_parameters("Class1")

        # if user selects train option, plot in train window                
        #                                                                    
        if window == igp.WINDOW_TRAIN:
            self.ui.input_display.clear_plot()
            self.data_points_t.set_two_ellipses(self.points, self.mu, self.cov)

        # otherwise, plot in eval window
        #
        else:
            self.ui.output_display.clear_plot()
            self.data_points_e.set_two_ellipses(self.points, self.mu, self.cov)

    # method: EventLoop::plot_four_ellipse
    #
    # arguments:
    #  window: the input window selected by the user (train or eval)
    #
    # This method plots the four ellipse preset pattern.
    #
    def plot_four_ellip(self, window):

        # retrieve user submitted parameters                       
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.menu_bar.clear_train_all.trigger()
        else:
            self.ui.menu_bar.clear_eval_all.trigger()
        self.retrieve_parameters()
        self.retrieve_class_parameters("Class0")
        self.retrieve_class_parameters("Class1")
        self.retrieve_class_parameters("Class2")
        self.retrieve_class_parameters("Class3")

        # if user selects train option, plot in train window                   
        #                                                                      
        if window == igp.WINDOW_TRAIN:
            self.ui.input_display.clear_plot()
            self.data_points_t.set_four_ellipses(self.points,
                                                 self.mu, self.cov)

        # otherwise plot in eval window
        #
        else:
            self.ui.output_display.clear_plot()
            self.data_points_e.set_four_ellipses(self.points,
                                                 self.mu, self.cov)

    # method: EventLoop::plot_rotated_ellips
    #
    # arguments:
    #  window: the input window selected by the user (train or eval)
    #
    # This method plots the rotated_ellips preset pattern.
    #
    def plot_rotated_ellips(self, window):

        # retrieve user defined parameters
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.menu_bar.clear_train_all.trigger()
        else:
            self.ui.menu_bar.clear_eval_all.trigger()
        self.retrieve_parameters()
        self.retrieve_class_parameters("Class0")
        self.retrieve_class_parameters("Class1")

        # if user selects train option, plot in train window                   
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.input_display.clear_plot()
            self.data_points_t.set_rotated_ellipse(self.points,
                                                  self.mu, self.cov)

        # otherwise plot in eval window
        #
        else:
            self.ui.output_display.clear_plot()
            self.data_points_e.set_rotated_ellipse(self.points,
                                                  self.mu, self.cov)

    # method: EventLoop::plot_toroidal
    #
    # arguments:
    #  window: the input window selected by the user (train or eval)
    #
    # This method plots the toroidal preset pattern.
    #
    def plot_toroidal(self, window):

        # retrieve user defined parameters
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.menu_bar.clear_train_all.trigger()
        else:
            self.ui.menu_bar.clear_eval_all.trigger()
        self.retrieve_parameters_mod()
        self.retrieve_class_parameters("Class0")
        self.retrieve_class_parameters("Class1")

        # if user selects train option, plot in train window                   
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.input_display.clear_plot()
            self.data_points_t.set_toroidal(self.params, self.mu, self.cov)

        # otherwise plot in eval window
        #
        else:
            self.ui.output_display.clear_plot()
            self.data_points_e.set_toroidal(self.params, self.mu, self.cov)

    # method: EventLoop::plot_ying_yang
    #
    # arguments:
    #  window: the input window selected by the user (train or eval)
    #
    # This method plots the ying yang preset pattern.
    #
    def plot_yin_yang(self, window):

        # retrieve user defined parameters
        #
        if window == igp.WINDOW_TRAIN:
            self.ui.menu_bar.clear_train_all.trigger()
        else:
            self.ui.menu_bar.clear_eval_all.trigger()
        self.retrieve_parameters_mod()
        self.retrieve_class_parameters("Class0")
        self.retrieve_class_parameters("Class1")

        # if user selects train option, plot in train window                   
        #
        if window == igp.WINDOW_TRAIN:
            #self.ui.input_display.clear_plot()
            self.data_points_t.set_yin_yang(self.params)

        # otherwise plot in eval window
        #
        else:
            #self.ui.output_display.clear_plot()
            self.data_points_e.set_yin_yang(self.params)
#------------------------------------------------------------------------------
#
# show pattern methods:
#   These methods use the sender method to determine the signal source, call
#   the igp.Second class from Parameters.py to open a *second window prompting users
#   to input parameters with defaults in place for each class and PyQt method
#   show() to bring up the parameter window for the user
#
#------------------------------------------------------------------------------

    def add_class_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Add Class")
        self.second.add_classes(sender)
        self.second.show()

    def two_gauss_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Two Gaussian (%s)" % sender.text())
        self.second.two_gaussian(sender.text())
        self.second.show()

    def four_gauss_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Four Gaussian (%s)" % sender.text())
        self.second.four_gaussian(sender.text())
        self.second.show()

    def over_gauss_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Over Gaussian (%s)" % sender.text())
        self.second.over_gaussian(sender.text())
        self.second.show()

    def two_ellipse_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Two Ellipses (%s)" % sender.text())
        self.second.two_ellipses(sender.text())
        self.second.show()

    def four_ellipse_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Four Ellipses (%s)" % sender.text())
        self.second.four_ellipses(sender.text())
        self.second.show()

    def rotated_ellipse_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Rotated Ellipses (%s)" % sender.text())
        self.second.rotated_ellipses(sender.text())
        self.second.show()

    def toroidal_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Toroidal (%s)" % sender.text())
        self.second.toroidal(sender.text())
        self.second.show()

    def yin_yang_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Yin-Yang (%s)" % sender.text())
        self.second.yin_yang(sender.text())
        self.second.show()

    # method: EventLoop::surface_color_show
    #
    # This method calls the second window for choosing the decision
    # surface color map
    #
    def surface_color_show(self):
        sender = self.sender()
        self.second = igp.Second(self)
        self.second.set_title("Pick a Surface Color Map")
        self.second.set_surface_color(sender)
        self.second.show()

    # method: EventLoop::set_plot_ranges
    #
    # This method calls the plot_ranges method which brings up a secondary
    # window to prompt user to input ranges for the X and Y axes of the Train
    # and Eval Windows and then makes this change in the UI.
    #
    def set_plot_ranges(self):

        self.settings = igp.Settings(self)

        # bring up pop-up window to get user input
        #
        self.settings.plot_ranges()

        # show new axes
        #
        self.settings.show()

    # method: EventLoop::prompt_set_gauss_prop
    #
    # This method calls the igp.Second.gausss_pattern method which brings up a secondary
    # window to prompt user to input cov of the Draw Gaussian
    #
    def prompt_set_gauss_prop(self):
        sender = self.sender()

        self.second = igp.Second(self)
        self.second.set_title("Gaussian Pattern(%s)" % sender.text())
        self.second.gauss_pattern(sender.text())
        self.second.show()

    def set_gauss_pattern(self):
        self.retrieve_parameters()

        # update cov in draw_gauss in both train and eval window
        #
        self.ui.input_display.cov = self.cov[0]
        self.ui.input_display.num_points = self.points

        self.ui.output_display.cov = self.cov[0]
        self.ui.output_display.num_points = self.points


    # method: EventLoop::change_range
    #
    # This method changes the ranges of the X and Y axes of the Train and Eval
    # windows to the users specifications
    #
    def change_range(self):

        ranges = np.empty((0, 0))

        # number of items in layout
        #
        index = self.settings.windowLayout.count() - 1

        for i in range(index):
            current_layout = self.settings.windowLayout.itemAt(i).layout()

            widget_count = current_layout.count()

            for widgets in range(widget_count):
                current_widget = current_layout.itemAt(widgets).widget()
                children = current_widget.findChildren(QtWidgets.QLineEdit)

                for params in range(len(children)):
                    try:
                        ranges = np.append(ranges, \
                                           float(children[params].text()))
                    except:
                        ranges = np.append(ranges, np.nan)

        ranges = np.reshape(ranges, (2, 2, 2))

        # sets range for train plot
        #
        if True not in np.isnan(ranges[0][0]):
            if ranges[0][0][0] > ranges[0][0][1]:
                ranges[0][0][0], ranges[0][0][1] = \
                    self.change_ranges_inverted_bound(ranges[0][0][0], ranges[0][0][1])
            self.ui.input_display.canvas.axes.set_xlim(ranges[0][0])
            self.ui.input_display.x_axis = np.linspace(ranges[0][0][0], ranges[0][0][1], 9)
            self.ui.input_display.canvas.axes.set_xticks(self.ui.input_display.x_axis)
            t_x = ranges[0][0]

        if True not in np.isnan(ranges[0][1]):
            if ranges[0][1][0] > ranges[0][1][1]:
                ranges[0][1][0], ranges[0][1][1] = \
                    self.change_ranges_inverted_bound(ranges[0][1][0], ranges[0][1][1])
            self.ui.input_display.canvas.axes.set_ylim(ranges[0][1])
            self.ui.input_display.y_axis = np.linspace(ranges[0][1][0], ranges[0][1][1], 9)
            self.ui.input_display.canvas.axes.set_yticks(self.ui.input_display.y_axis)
            t_y = ranges[0][1]

        # sets range for eval plot
        #
        if True not in np.isnan(ranges[1][0]):
            if ranges[1][0][0] > ranges[1][0][1]:
                ranges[1][0][0], ranges[1][0][1] = \
                    self.change_ranges_inverted_bound(ranges[1][0][0], ranges[1][0][1])
            self.ui.output_display.canvas.axes.set_xlim(ranges[1][0])
            self.ui.output_display.x_axis = np.linspace(ranges[1][0][0], ranges[1][0][1], 9)
            self.ui.output_display.canvas.axes.set_xticks(self.ui.output_display.x_axis)
            e_x = ranges[1][0]

        if True not in np.isnan(ranges[1][1]):
            if ranges[1][1][0] > ranges[1][1][1]:
                ranges[1][1][0], ranges[1][1][1] = \
                    self.change_ranges_inverted_bound(ranges[1][1][0], ranges[1][1][1])
            self.ui.output_display.canvas.axes.set_ylim(ranges[1][1])
            self.ui.output_display.y_axis = np.linspace(ranges[1][1][0], ranges[1][1][1], 9)
            self.ui.output_display.canvas.axes.set_yticks(self.ui.output_display.y_axis)
            e_y = ranges[1][1]
        self.ui.input_display.canvas.draw_idle()
        self.ui.output_display.canvas.draw_idle()

    def change_ranges_inverted_bound(self, min, max):
        temp = min
        min = max
        max = temp
        message = "Invalid entry: Maximum bound was less than Minimum bound " \
                  "\n\nBounds were reversed to keep Minimum less than Maximum"
        self.display_warning(message)
        return min, max

    def algorithm_pca_cd(self):

        # set up the algorithm
        #
        self.algo = AlgorithmPCA(self.ui.input_display, \
                                 self.ui.output_display,\
                                 self.ui.process_desc.output,"CD")
        self.ui.process_desc.output.append("Algorithm: " +
                                           igw.ALG_PCA_CD)

        # initialize the model with GUI and the algorithm
        self.model = model.Model(self.algo,self.ui.input_display, self.ui.output_display,self.ui.process_desc, self.is_normalized)

        self.initialized = False

    def algorithm_pca_ci(self):
        # set up the algorithm
        #
        self.algo = AlgorithmPCA(self.ui.input_display,\
                                  self.ui.output_display,\
                                  self.ui.process_desc.output,"CI")

        self.ui.process_desc.output.append("Algorithm:  " +
                                           igw.ALG_PCA_CI)

        self.initialized = False
        # set up the model with the GUI and the algorithm
        self.model = model.Model(self.algo,self.ui.input_display, self.ui.output_display,self.ui.process_desc, self.is_normalized)


    def algorithm_svm(self):

        # retrieve some params
        #
        maxiter = int(self.second.svm_max_iter_line.text())
        gamma = self.second.svm_gamma_selection.currentText()

        self.algo = AlgorithmSVM(self.ui.input_display,
                                 self.ui.output_display,
                                 self.ui.process_desc.output,
                                 maxiter, gamma)

        self.ui.process_desc.output.append("Algorithm:  " +
                                           igw.ALG_SUPPORT)

        self.initialized = False
        self.model = model.Model(self.algo, self.ui.input_display,
                                self.ui.output_display,
                                self.ui.process_desc, self.is_normalized)

    def prompt_algo_svm(self):
        sender = self.sender()

        self.second = igp.Second(self)
        self.second.set_title("%s" % sender.text())
        self.second.svm_params(sender.text())
        self.second.show()

    def algorithm_knn(self):
        
        # retrieve some params
        #
        weight = self.second.knn_weight_selection.currentText()
        algo = self.second.knn_algo_selection.currentText()
        neighbors = self.second.knn_neighbor_line.text()

        for i in self.ui.input_display.class_info:
            if int(neighbors)*2 > len(self.ui.input_display.class_info[i][2]):
                self.ui.process_desc.output.append("Data not large enough for Knn")
                return

        self.algo = AlgorithmKNN(self.ui.input_display,
                                self.ui.output_display,
                                self.ui.process_desc,
                                neighbors,
                                algo,
                                weight)
        self.ui.process_desc.output.append("Algorithm:  " +
                                           igw.ALG_NEAREST)

        self.initialized = False
        self.model = model.Model(self.algo, self.ui.input_display,
                                 self.ui.output_display,
                                 self.ui.process_desc, self.is_normalized)

    def prompt_algo_knn(self):
        sender = self.sender()

        self.second = igp.Second(self)
        self.second.set_title("%s" % sender.text())
        self.second.knn_params(sender.text())
        self.second.show()

    def algorithm_mlp(self):

        # retrieve some params
        #
        nlayers = self.second.mlp_layer_line.text()
        nlayers = nlayers.replace("(", "").replace(")", "").replace(",", " ")
        nlayers = np.fromstring(nlayers, sep=" ", dtype=int)

        solver = self.second.mlp_solver_selection.currentText()

        self.algo = AlgorithmMLP(self.ui.input_display,\
                                 self.ui.output_display,\
                                 self.ui.process_desc.output, nlayers, solver)
        self.ui.process_desc.output.append("Algorithm: "+
                                           igw.ALG_MLP)
        self.initialized = False
        self.model = model.Model(self.algo, self.ui.input_display,
                                 self.ui.output_display,
                                 self.ui.process_desc,
                                 self.is_normalized)

    def prompt_algo_mlp(self):
        sender = self.sender()

        self.second = igp.Second(self)
        self.second.set_title("%s" % sender.text())
        self.second.mlp_params(sender.text())
        self.second.show()

    def algorithm_kmeans(self):

        # retrieve some params
        #
        n_clusters = int(self.second.kmeans_ncluster_line.text())
        init = self.second.kmeans_init_selection.currentText()
        n_init = int(self.second.kmeans_ninit_line.text())
        max_iter = int(self.second.kmeans_max_iter_line.text())


        for i in self.ui.input_display.class_info:
            if n_clusters > len(self.ui.input_display.class_info[i][2]):
                self.ui.process_desc.output.append("Data not large enough for Kmeans")
                return

        self.algo = AlgorithmKMeans(self.ui.input_display,\
                                    self.ui.output_display,\
                                    self.ui.process_desc.output, n_clusters,
                                    init, n_init, max_iter)
        self.ui.process_desc.output.append("Algorithm: " +
                                           igw.ALG_KMEANS)
        self.initialized = False

        self.model = model.Model(self.algo, self.ui.input_display,
                                self.ui.output_display,
                                self.ui.process_desc,
                                self.is_normalized)

    def prompt_algo_kmeans(self):
        sender = self.sender()

        self.second = igp.Second(self)
        self.second.set_title("%s" % sender.text())
        self.second.kmeans_params(sender.text())
        self.second.show()
        
    def algorithm_lda_cd(self):

        # set up the algorithm
        #
        self.algo = AlgorithmLDA(self.ui.input_display, \
                                 self.ui.output_display, \
                                 self.ui.process_desc.output, "CD")
        self.ui.process_desc.output.append("Algorithm: " +
                                           igw.ALG_LDA_CD)

        # initialize the model with GUI and the algorithm
        self.model = model.Model(self.algo, self.ui.input_display, 
                        self.ui.output_display, self.ui.process_desc,
                        self.is_normalized)

        self.initialized = False

    def algorithm_lda_ci(self):
        # set up the algorithm
        #
        self.algo = AlgorithmLDA(self.ui.input_display, \
                                 self.ui.output_display, \
                                 self.ui.process_desc.output, "CI")
        self.ui.process_desc.output.append("Algorithm: " +
                                           igw.ALG_LDA_CI)

        # initialize the model with GUI and the algorithm
        self.model = model.Model(self.algo, self.ui.input_display, 
                        self.ui.output_display, self.ui.process_desc, self.is_normalized)

        self.initialized = False


    def algorithm_rf(self):

        # retrieve some params
        #
        n_estimators = int(self.second.rf_ntree_line.text())
        crit = self.second.rf_crit_selection.currentText()



        self.algo = AlgorithmRF(self.ui.input_display,
                                self.ui.output_display,
                                self.ui.process_desc.output, n_estimators,
                                crit)
        self.ui.process_desc.output.append("Algorithm: "+
                                           igw.ALG_RF)
        self.initialized = False
        self.model = model.Model(self.algo, self.ui.input_display,
                                 self.ui.output_display,
                                 self.ui.process_desc,
                                 self.is_normalized)

    def prompt_algo_rf(self):
        sender = self.sender()

        self.second = igp.Second(self)
        self.second.set_title("%s" % sender.text())
        self.second.rf_params(sender.text())
        self.second.show()
    # method: EventLoop::run_complete
    #
    # This method is called when the user selects Process > Run... and will
    # execute all steps of the selected algorithm.
    #
    def run_complete(self):

        # run the model until last step is triggered
        #
        if self.algo is not None:
            if not self.initialized:
                self.run_initialize()
                self.initialized = True

            if self.data:
                while not self.model.is_done():
                    self.run_next()

                self.model.reset_step()
                self.initialized = False

            else:
                self.model.reset_step()
                self.initialized = False

        # if user runs without selecting an algorithm first,
        # print message in the process description window
        #
        else:
            self.ui.process_desc.output.append("Status: No algorithm is \
currently selected.")

    # method: EventLoop::run_initialize
    #
    # Initialize the model and checks if there is data available
    #
    def run_initialize(self):
        if self.algo is not None:
            self.value = 0
            self.ui.process_desc.pbar.setValue(self.value)

            # check available data
            #
            if not self.model.prepare_data(self.ui.input_display.class_info):
                self.data = False
                self.ui.process_desc.output.append("Status: There is no data \
                to initialize.")
                self.initialized = False
                return False

            # trigger data and initialize flag
            #
            else:
                self.data = True
                self.initialized = True


        else:
            return False

    # method: EventLoop::run_step
    # this method is called when the user selects Process > Step and will    
    # execute a single step of the selected algorithm
    #
    def run_step(self):
        #QtWidgets.QApplication.processEvents()
        if self.algo is not None:
            if not self.initialized:
                self.run_initialize()
                self.ui.process_desc.output.append("\nAlgorithm Initialized, you may proceed\n")
                self.initialized = True

            # if the algorithm is complete, reset the "step" to zero but
            # if not, execute the next step
            #
            else:
                if self.data:
                    if not self.model.is_done():
                        self.run_next()

                    else:
                        self.model.reset_step()
                        self.initialized = False


        # if user steps without selecting an algorithm first,
        # print out message in the process description window
        # 
        else:
            self.ui.process_desc.output.append("Status: No algorithm is currently selected.")

    # method: EventLoop::run_next
    #
    # this method executes the next step of an algorithm 
    #
    def run_next(self):
        if self.algo is not None:
            if not self.data:

                # display message in process description window
                #
                self.ui.process_desc.output.append("Status: There is no data to proceed with.")
            else:
                self.model.increment_step()
                self.model.step()


    # method: EventLoop::set_point
    #
    # this method enables the user to draw points on the Train and Eval windows
    #
    def set_point(self):

        # remove checkmark from Draw Gaussian if it's checked
        #
        self.ui.menu_bar.draw_points_menu.setChecked(True)
        self.ui.menu_bar.draw_gauss_menu.setChecked(False)

        self.ui.input_display.set_point()
        self.ui.output_display.set_point()


        # if user tries to draw before selecting a class raise an error
        #
        if self.classes == None:

            # define message to display to user
            #
            message = "Warning: Please select an input class."

            # display warning message in process description window
            #
            self.display_warning(message)

    # method: EventLoop::set_gauss
    #
    # this method enables the user to draw gaussian plots on the Train and Eval windows
    #
    def set_gauss(self):

        # remove checkmark from Draw Points if it's checked
        #
        self.ui.menu_bar.draw_points_menu.setChecked(False)
        self.ui.menu_bar.draw_gauss_menu.setChecked(True)

        self.ui.input_display.set_gauss()
        self.ui.output_display.set_gauss()

        # if user tries to draw before selecting a class raise an error
        #
        if self.classes == None:

            # define message to display to user
            #
            message = "Warning: Please select an input class."

            # display message in process description window
            #
            self.display_warning(message)

    # method: EventLoop::normalize_data
    #
    # This method calls the process of Normalize Data in train and 
    # eval window
    #
    def normalize_data(self):

        # if the 'Normalize Data' is checked
        #
        if self.ui.menu_bar.set_normalize_menu.isChecked():
            self.is_normalized = True
            self.ui.process_desc.output.append("Select Normalize Data")

        else:
            self.is_normalized = True
            self.ui.process_desc.output.append("Unselect Normalize Data")


    # method: EventLoop::retrieve_parameters
    #
    # This method extracts the parameters from the input window that appears
    #  when a pattern is selected.
    #
    def retrieve_parameters(self):
        # set up mean, covariance and points variables
        #
        self.mu = np.nan
        self.cov = np.nan
        self.points = np.nan

        # find how many child widgets there are in the main GUI
        #
        index = self.second.windowLayout.count() - 1

        # set up the mean and covariance depending on the index
        #
        mu = np.empty((index - 1, 2, 1)) * np.nan
        cov = np.empty((index - 1, 2, 2)) * np.nan


        # iterate through the child widgets to find number of inputs
        #
        for i in range(index):

            current_layout = self.second.windowLayout.itemAt(i).layout()

            if type(current_layout) == QtWidgets.QHBoxLayout:
                
                widget_count = current_layout.count()

                # iterate through widgets to find the extracted info
                #
                for widgets in range(widget_count):
                    current_widget = current_layout.itemAt(widgets).widget()
                    children = current_widget.findChildren(QtWidgets.QLineEdit)

                    if len(children)==1:

                        try:
                            self.points = int(self.second.num_size.text())
                            continue
                        except:
                            try:
                                # get number of points from Set Gaussian
                                #
                                self.points = int(self.second.num_points.text())
                            except:
                                print("Errors: Cannot set number of points")

                    # if there are two input boxes, try mean
                    #
                    if len(children) == 2:
                        try:
                            for extract_mu in range(len(children)):
                                mu[i - 1][extract_mu] = \
                                    float(children[extract_mu].text())
                        except:
                            pass

                    # 4 boxes are for covariance matrices
                    #
                    elif len(children) == 4:
                        temp = np.empty((0, 0))
                        for extract_cov in range(len(children)):
                            try:
                                temp = np.append(temp,\
                                        float(children[extract_cov].text()))
                            except:
                                pass
                        try:
                            cov[i - 1] = np.reshape(temp, (-1, 2))
                        except:
                            pass
        self.mu = mu
        self.cov = cov
        self.mu = self.mu.reshape(len(self.mu), -1)

    # method: EventLoop::retrieve_parameters_tor
    #
    # This method retrieves parameters for the toroidal pattern
    #
    def retrieve_parameters_mod(self):

        # declare local variables
        #
        self.mu = np.nan
        self.cov = np.nan
        self.points = np.nan

        params = np.empty([0, 0])
        index = self.second.windowLayout.count() - 1

        mu = np.empty((2, 1)) * np.nan
        cov = np.empty((2, 2)) * np.nan

        for i in range(index):
            current_layout = self.second.windowLayout.itemAt(i).layout()

            if type(current_layout) == QtWidgets.QHBoxLayout:
                widget_count = current_layout.count()

                for widgets in range(widget_count):
                    current_widget = current_layout.itemAt(widgets).widget()
                    children = current_widget.findChildren(QtWidgets.QLineEdit)

                    # check if the paramters is not 4 for the number of points, and radius inputs
                    #
                    if params.size != 4:
                        try:
                            for extract in range(len(children)):
                                if children[extract].text():
                                    params = np.append(params,\
                                            float(children[extract].text()))
                                else:
                                    params = np.append(params, np.nan)
                            continue
                        except:
                            pass

                    if len(children) == 2:
                        try:
                            for extract_mu in range(len(children)):
                                mu[extract_mu][0] = \
                                    float(children[extract_mu].text())
                        except:
                            pass

                    elif len(children) == 4:
                        temp = np.empty((0, 0))
                        for extract_cov in range(len(children)):
                            try:
                                temp = np.append(temp, \
                                        float(children[extract_cov].text()))
                            except:
                                pass
                        try:
                            if temp.size == 4:
                                cov = np.reshape(temp, (-1, 2))
                            else:
                                cov = np.empty((2, 2)) * np.nan
                        except:
                            pass

        self.mu = mu
        self.cov = cov
        self.params = params

#
# end of class

#
# end of file
