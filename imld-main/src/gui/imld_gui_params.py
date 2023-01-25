#!/usr/bin/env python3
#
# file: imld_params
#
# This script enables the parameters of each algorithm to display when one
#  is selected. These parameters vary based on the Algorithm selected.
#------------------------------------------------------------------------------

# import system modules
#
import os
# import modules for widget-based UI
#
from PyQt5 import QtCore, QtGui, QtWidgets
import gui.imld_gui_window as igw
from matplotlib.colors import cnames

import lib.imld_constants_datagen as icd


#-----------------------------------------------------------------------------
# global variables defined here
#-----------------------------------------------------------------------------

WINDOW_TRAIN = "Train"
WINDOW_EVAL = "Eval"


KNN_WEIGHT = ["uniform", "distance"]
KNN_ALGO = ["auto", "ball_tree", "kd_tree", "brute"]
SVM_MAXITER = ["-1", "1"]
SVM_GAMMA = ["auto", "scale"]
KMEANS_INIT = ['k-means++', 'random']
MLP_SOLVER = ['lbfgs', 'sgd', 'adam']
RF_CRIT = ['gini', 'entropy']

# class: Second
#
# arguments:
#  QtWidgets.QDialog: Qt base class of dialog widgets  
#
# This class is presumably named for the second pop-up window that
# allows the input of parameters when a pattern is chosen 
#
class Second(QtWidgets.QDialog):

    # method: Second::constructor
    #
    # arguments:
    #  event_loop: the event loop
    #
    def __init__(self, event_loop):
        super().__init__()
        self.event_loop = event_loop
        self.title = None

        self.width = 320
        self.height = 100
  
        # layout for points input
        #
        self.num_points_box = QtWidgets.QGroupBox("Number of Points")
        self.size_of_gauss_box = QtWidgets.QGroupBox("Size of Gaussian")
        self.num_points_box_t = QtWidgets.QGroupBox("Number of Points (Ring)")
        self.inner_and_outer_radius = QtWidgets.QGroupBox("Inner and Outer Radius (Ring)")
        self.num_points_yin_box = QtWidgets.QGroupBox("Number of Points (Yin)")
        self.num_points_yang_box = QtWidgets.QGroupBox("Number of Points (Yang)")

        # layouts for the first gaussian inputs
        #
        self.horizontalGroupBox = QtWidgets.QGroupBox("Mu (First)")
        self.horizontalGroupBox1 = QtWidgets.QGroupBox("Cov (First)")

        # layouts for the second gaussian inputs
        #
        self.horizontalGroupBox2 = QtWidgets.QGroupBox("Mu (Second)")
        self.horizontalGroupBox3 = QtWidgets.QGroupBox("Cov (Second)")

        # layouts for the third gaussian inputs
        #
        self.horizontalGroupBox4 = QtWidgets.QGroupBox("Mu (Third)")
        self.horizontalGroupBox5 = QtWidgets.QGroupBox("Cov (Third)")

        # layouts for the fourth gaussian inputs
        #
        self.horizontalGroupBox6 = QtWidgets.QGroupBox("Mu (Fourth)")
        self.horizontalGroupBox7 = QtWidgets.QGroupBox("Cov (Fourth)")

        # layouts for text input (mu and cov)
        #
        self.layout = QtWidgets.QGridLayout()
        self.layout1 = QtWidgets.QGridLayout()
        self.layout2 = QtWidgets.QGridLayout()
        self.layout3 = QtWidgets.QGridLayout()
        self.layout4 = QtWidgets.QGridLayout()
        self.layout5 = QtWidgets.QGridLayout()
        self.layout6 = QtWidgets.QGridLayout()
        self.layout7 = QtWidgets.QGridLayout()
        self.layout8 = QtWidgets.QGridLayout()
        self.layout9 = QtWidgets.QGridLayout()
        
        # create QLineEdit instances to hold parameters to be
        # inserted into pop-up param windows
        #
        self.num_points = QtWidgets.QLineEdit()
        self.num_size = QtWidgets.QLineEdit()
        self.num_points_yin = QtWidgets.QLineEdit()
        self.num_points_yang = QtWidgets.QLineEdit()

        self.add_class = QtWidgets.QLineEdit()

        self.surface_color = QtWidgets.QLineEdit()

        self.num_points_tor = QtWidgets.QLineEdit()
        self.inner_ring = QtWidgets.QLineEdit()
        self.outer_ring = QtWidgets.QLineEdit()

        self.mu_0 = QtWidgets.QLineEdit()
        self.mu_1 = QtWidgets.QLineEdit()
        self.mu_2 = QtWidgets.QLineEdit()
        self.mu_3 = QtWidgets.QLineEdit()

        self.mu_4 = QtWidgets.QLineEdit()
        self.mu_5 = QtWidgets.QLineEdit()
        self.mu_6 = QtWidgets.QLineEdit()
        self.mu_7 = QtWidgets.QLineEdit()

        self.cov_0 = QtWidgets.QLineEdit()
        self.cov_1 = QtWidgets.QLineEdit()
        self.cov_2 = QtWidgets.QLineEdit()
        self.cov_3 = QtWidgets.QLineEdit()
        self.cov_4 = QtWidgets.QLineEdit()
        self.cov_5 = QtWidgets.QLineEdit()
        self.cov_6 = QtWidgets.QLineEdit()
        self.cov_7 = QtWidgets.QLineEdit()

        self.cov_8 = QtWidgets.QLineEdit()
        self.cov_9 = QtWidgets.QLineEdit()
        self.cov_10 = QtWidgets.QLineEdit()
        self.cov_11 = QtWidgets.QLineEdit()
        self.cov_12 = QtWidgets.QLineEdit()
        self.cov_13 = QtWidgets.QLineEdit()
        self.cov_14 = QtWidgets.QLineEdit()
        self.cov_15 = QtWidgets.QLineEdit()

        # layouts for the KNN params
        #
        self.knn_neighbor_box = QtWidgets.QGroupBox("K Neighbor: Number of neighbors")
        self.knn_neighbor_line = QtWidgets.QLineEdit()

        self.knn_algo_box = QtWidgets.QGroupBox("Algorithm: Algorithm used to compute the nearest neighbors")
        self.knn_weights_box = QtWidgets.QGroupBox("Weights: Weight function used in prediction")

        # layout for SVM params
        #
        self.svm_max_iter_box = QtWidgets.QGroupBox("Max Iter")
        self.svm_max_iter_line = QtWidgets.QLineEdit()
        self.svm_gamma_box = QtWidgets.QGroupBox("Gamma")

        # layout for KMeans params
        #
        self.kmeans_ncluster_box = QtWidgets.QGroupBox("Number of clusters")
        self.kmeans_ncluster_line = QtWidgets.QLineEdit()

        self.kmeans_init_box = QtWidgets.QGroupBox("Method of initialization")
        self.kmeans_ninit_box = QtWidgets.QGroupBox("Number of time K-means algorithm will be run with different centroid seeds")
        self.kmeans_ninit_line = QtWidgets.QLineEdit()
        self.kmeans_max_iter_box = QtWidgets.QGroupBox("Maximum number of iterations of the k-means algorithm for a single run")
        self.kmeans_max_iter_line = QtWidgets.QLineEdit()

        # layout for MLP params
        #
        self.mlp_layer_box = QtWidgets.QGroupBox("Number of hidden layers")
        self.mlp_layer_line = QtWidgets.QLineEdit()

        self.mlp_solver_box = QtWidgets.QGroupBox("The solver for weight optimization")
        
        # layout for RF params
        #
        self.rf_ntree_box = QtWidgets.QGroupBox("Number of trees")
        self.rf_ntree_line = QtWidgets.QLineEdit()

        self.rf_crit_box = QtWidgets.QGroupBox("The function to measure the quality of a split")

        self.initUI()

    # method: Second::initUI
    #
    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(self.width, self.height)

        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep + 'logo.png'))

        # set Validator
        #
        self.num_points.setValidator(QtGui.QIntValidator())
        self.num_size.setValidator(QtGui.QIntValidator())
        self.num_points_yin.setValidator(QtGui.QIntValidator())
        self.num_points_yang.setValidator(QtGui.QIntValidator())
                
        self.mu_0.setValidator(QtGui.QDoubleValidator())
        self.mu_1.setValidator(QtGui.QDoubleValidator())
        self.mu_2.setValidator(QtGui.QDoubleValidator())
        self.mu_3.setValidator(QtGui.QDoubleValidator())

        self.cov_0.setValidator(QtGui.QDoubleValidator())
        self.cov_1.setValidator(QtGui.QDoubleValidator())
        self.cov_2.setValidator(QtGui.QDoubleValidator())
        self.cov_3.setValidator(QtGui.QDoubleValidator())
        self.cov_4.setValidator(QtGui.QDoubleValidator())
        self.cov_5.setValidator(QtGui.QDoubleValidator())
        self.cov_6.setValidator(QtGui.QDoubleValidator())
        self.cov_7.setValidator(QtGui.QDoubleValidator())

    # method: Second::set_title
    # arguments: None
    # This method sets the title of the secondary windows
    #
    def set_title(self,title):
        self.title = title
        self.setWindowTitle(self.title)

    # method: Second::set_color
    # arguments: None
    # This method sets the drop-down list for selectable colors
    #
    def set_color(self):
        self.p_color = QtWidgets.QComboBox(self)
        self.p_color.setStyleSheet("QComboBox { combobox-popup: 0; }")
        self.p_color.setMaxVisibleItems(6)
        self.p_color.addItem("color")
        self.p_color.addItems([*cnames])

    # method: Second::set_surface_color
    # arguments:
    #  sender: the signal sourced from GUI button "Set Colors"
    # This method sets the title of the secondary windows
    #
    def set_surface_color(self, sender):

        # set place holder text to show the user what will be used if left blank
        #
        self.surface_color.setPlaceholderText('winter')

        # add to specific second window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.surface_color)
        self.windowLayout.addLayout(testLayout)

        # set up button for submission
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)
        self.setLayout(self.windowLayout)

        # if button is pushed, set up the colormap for the decision surface and reset the text
        #
        button.clicked.connect(lambda : self.event_loop.handled_surface_color(self.surface_color.text()))
        button.clicked.connect(lambda: self.surface_color.setText(""))
        self.setLayout(self.windowLayout)

    # method: Second::add_classes
    #
    # arguments: None
    #
    # This method pulls up the parameter window
    # when the user selects the "Add Class" from the menu.
    #
    def add_classes(self, sender):

        self.add_class.setPlaceholderText("Add_Class")
        self.windowLayout = QtWidgets.QVBoxLayout()
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.add_class)
        self.windowLayout.addLayout(testLayout)
        # reset colors chosen after submission
        self.event_loop.ui.input_display_color_c = None
        self.event_loop.ui.output_display_color_c = None

        # adds the color box to the second window
        #
        self.set_color()
        testLayout.addWidget(self.p_color)
        self.windowLayout.addLayout(testLayout)

        # set up button for submission
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)
        self.setLayout(self.windowLayout)

        # use the selected color for plotting
        #
        self.p_color.setCurrentIndex(0)

        self.p_color.currentIndexChanged[str].connect(self.event_loop.handled_color)
        button.clicked.connect\
            (lambda: self.event_loop.handled_color_saver())
        # if button is pushed add class to the both menu bar and storage dictionary
        button.clicked.connect\
            (lambda: self.event_loop.retrieve_class_parameters(self.add_class.text()))

        button.clicked.connect(lambda: self.add_class.setText(""))
        button.clicked.connect(lambda: self.close())

    # method: Second::two_gaussian
    #
    # arguments:
    #  sender: title of button pushed that invoked this method
    #   
    def gauss_pattern(self, sender):
        
        self.create_grid_layout_one()

        # create text for all text box
        #
        self.num_size.setText("25")

        self.cov_0.setText("0.0500")
        self.cov_1.setText("0.0000")
        self.cov_2.setText("0.0000")
        self.cov_3.setText("0.0500")

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.size_of_gauss_box)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox1)
        self.windowLayout.addLayout(testLayout)


        # set up button for submission
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)
        
        button.clicked.connect\
            (lambda:self.event_loop.set_gauss_pattern())
        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    # method: Second::two_gaussian
    #
    # arguments:
    #  sender: title of button pushed that invoked this method
    #
    # This method pulls up the parameter window with default values
    # when the user selects the "Two Gaussian" pattern from the menu. 
    #   
    def two_gaussian(self, sender):
        self.create_grid_layout_two()

        # create text for all text box
        #
        self.num_points.setText(str(icd.DEFAULT_NPTS_PER_CLASS))

        self.mu_0.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_MEAN[0][0]))
        self.mu_1.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_MEAN[0][1]))

        self.mu_2.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_MEAN[1][0]))
        self.mu_3.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_MEAN[1][1]))

        self.cov_0.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_COV[0][0][0]))
        self.cov_1.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_COV[0][0][1]))
        self.cov_2.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_COV[0][1][0]))
        self.cov_3.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_COV[0][1][1]))

        self.cov_4.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_COV[1][0][0]))
        self.cov_5.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_COV[1][0][1]))
        self.cov_6.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_COV[1][1][0]))
        self.cov_7.setText("{0:.4f}".format(icd.DEFAULT_TWOGAUSSIAN_COV[1][1][1]))

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_box)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox)
        testLayout.addWidget(self.horizontalGroupBox1)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox2)
        testLayout.addWidget(self.horizontalGroupBox3)
        self.windowLayout.addLayout(testLayout)

        # set up button for submission
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        # if user selects train option, plot in train window
        #  note that the lambda function creates a "callable" object
        #  which is required by connect()
        #
        if sender == WINDOW_TRAIN:
            button.clicked.connect\
                (lambda:self.event_loop.plot_two_gaus(WINDOW_TRAIN))

        # otherwise, plot in eval window
        #
        elif sender == WINDOW_EVAL:
            button.clicked.connect\
                (lambda:self.event_loop.plot_two_gaus(WINDOW_EVAL))

        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    # method: Second::four_gaussian                                 
    #
    # arguments:
    #  sender: title of button pushed that invoked this method                
    #
    # This method pulls up the parameter window with default values
    # when the user selects the "Four Gaussian" pattern from the menu. 
    #   
    def four_gaussian(self, sender):
        self.create_grid_layout_four()

        # create plaecholder text for all text box
        #
        self.num_points.setText(str(icd.DEFAULT_NPTS_PER_CLASS))

        self.mu_0.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_MEAN[0][0]))
        self.mu_1.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_MEAN[0][1]))

        self.mu_2.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_MEAN[1][0]))
        self.mu_3.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_MEAN[1][1]))

        self.cov_0.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[0][0][0]))
        self.cov_1.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[0][0][1]))
        self.cov_2.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[0][1][0]))
        self.cov_3.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[0][1][1]))

        self.cov_4.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[1][0][0]))
        self.cov_5.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[1][0][1]))
        self.cov_6.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[1][1][0]))
        self.cov_7.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[1][1][1]))

        self.mu_4.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_MEAN[2][0]))
        self.mu_5.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_MEAN[2][1]))

        self.mu_6.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_MEAN[3][0]))
        self.mu_7.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_MEAN[3][1]))

        self.cov_8.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[2][0][0]))
        self.cov_9.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[2][0][1]))
        self.cov_10.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[2][1][0]))
        self.cov_11.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[2][1][1]))

        self.cov_12.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[3][0][0]))
        self.cov_13.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[3][0][1]))
        self.cov_14.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[3][1][0]))
        self.cov_15.setText("{0:.4f}".format(icd.DEFAULT_FOURGAUSSIAN_COV[3][1][1]))

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_box)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox)
        testLayout.addWidget(self.horizontalGroupBox1)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox2)
        testLayout.addWidget(self.horizontalGroupBox3)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox4)
        testLayout.addWidget(self.horizontalGroupBox5)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox6)
        testLayout.addWidget(self.horizontalGroupBox7)
        self.windowLayout.addLayout(testLayout)

        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        # if user selects train option, plot in train window
        #  note that the lambda function creates a "callable" object
        #  which is required by connect()
        #
        if sender == WINDOW_TRAIN:
            button.clicked.connect\
                (lambda:self.event_loop.plot_four_gaus(WINDOW_TRAIN))

        # otherwise, plot in eval window
        #
        elif sender == WINDOW_EVAL:
            button.clicked.connect\
                (lambda:self.event_loop.plot_four_gaus(WINDOW_EVAL))

        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    # method: Second::over_gaussian
    #
    # arguments:
    #  sender: title of button pushed that invoked this method
    #
    # This method holds the parameters for the "Overlapping Gaussian"
    #  pattern.
    #
    def over_gaussian(self, sender):
        self.create_grid_layout_two()

        # create placeholder text for all text box
        #
        self.num_points.setText(str(icd.DEFAULT_NPTS_PER_CLASS))

        self.mu_0.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_MEAN[0][0]))
        self.mu_1.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_MEAN[0][1]))

        self.mu_2.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_MEAN[1][0]))
        self.mu_3.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_MEAN[1][1]))

        self.cov_0.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_COV[0][0][0]))
        self.cov_1.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_COV[0][0][1]))
        self.cov_2.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_COV[0][1][0]))
        self.cov_3.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_COV[0][1][1]))

        self.cov_4.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_COV[1][0][0]))
        self.cov_5.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_COV[1][0][1]))
        self.cov_6.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_COV[1][1][0]))
        self.cov_7.setText("{0:.4f}".format(icd.DEFAULT_OVERLAPGAUSSIAN_COV[1][1][1]))

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_box)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox)
        testLayout.addWidget(self.horizontalGroupBox1)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox2)
        testLayout.addWidget(self.horizontalGroupBox3)
        self.windowLayout.addLayout(testLayout)

        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        # if train button selected, plot in train window
        #  note that the lambda function creates a "callable" object      
        #  which is required by connect()    
        #
        if sender == WINDOW_TRAIN:
            button.clicked.connect\
                (lambda:self.event_loop.plot_ovlp_gaussian(WINDOW_TRAIN))

        # otherwise, plot in eval window
        #
        elif sender == WINDOW_EVAL:
            button.clicked.connect\
                (lambda:self.event_loop.plot_ovlp_gaussian(WINDOW_EVAL))

        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    # method: Second::two_ellipses
    #
    # arguments:
    #  sender: title of button pushed that invoked this method
    #
    # This method creates the parameter window for the "Two Ellipses" pattern
    #  option.
    #
    def two_ellipses(self, sender):
        self.create_grid_layout_two()

        # create placeholder text for all text box
        #
        self.num_points.setText(str(icd.DEFAULT_NPTS_PER_CLASS))

        self.mu_0.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_MEAN[0][0]))
        self.mu_1.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_MEAN[0][1]))

        self.mu_2.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_MEAN[1][0]))
        self.mu_3.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_MEAN[1][1]))

        self.cov_0.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_COV[0][0][0]))
        self.cov_1.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_COV[0][0][1]))
        self.cov_2.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_COV[0][1][0]))
        self.cov_3.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_COV[0][1][1]))

        self.cov_4.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_COV[1][0][0]))
        self.cov_5.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_COV[1][0][1]))
        self.cov_6.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_COV[1][1][0]))
        self.cov_7.setText("{0:.4f}".format(icd.DEFAULT_TWOELLIPSE_COV[1][1][1]))

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_box)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox)
        testLayout.addWidget(self.horizontalGroupBox1)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox2)
        testLayout.addWidget(self.horizontalGroupBox3)
        self.windowLayout.addLayout(testLayout)

        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        # if train button selected, plot in train window              
        #  note that the lambda function creates a "callable" object        
        #  which is required by connect()                                   
        #                                                                    
        if sender == WINDOW_TRAIN:
            button.clicked.connect\
                (lambda:self.event_loop.plot_two_ellip(WINDOW_TRAIN))

        # otherwise, plot in eval window
        #
        elif sender == WINDOW_EVAL:
            button.clicked.connect\
                (lambda:self.event_loop.plot_two_ellip(WINDOW_EVAL))

        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    # method: Second::four_ellipses
    #
    # arguments:
    #  sender: title of button pushed that invoked this method
    #
    # This method creates the parameter window that pops up when the
    #  "Four Ellipses" option is selected.
    #
    def four_ellipses(self,sender):
        self.create_grid_layout_four()

        # create plaecholder text for all text box
        #
        self.num_points.setText(str(icd.DEFAULT_NPTS_PER_CLASS))
        self.mu_0.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_MEAN[0][0]))
        self.mu_1.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_MEAN[0][1]))

        self.mu_2.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_MEAN[1][0]))
        self.mu_3.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_MEAN[1][1]))

        self.cov_0.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[0][0][0]))
        self.cov_1.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[0][0][1]))
        self.cov_2.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[0][1][0]))
        self.cov_3.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[0][1][1]))

        self.cov_4.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[1][0][0]))
        self.cov_5.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[1][0][1]))
        self.cov_6.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[1][1][0]))
        self.cov_7.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[1][1][1]))

        self.mu_4.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_MEAN[2][0]))
        self.mu_5.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_MEAN[2][1]))

        self.mu_6.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_MEAN[3][0]))
        self.mu_7.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_MEAN[3][1]))

        self.cov_8.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[2][0][0]))
        self.cov_9.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[2][0][1]))
        self.cov_10.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[2][1][0]))
        self.cov_11.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[2][1][1]))

        self.cov_12.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[3][0][0]))
        self.cov_13.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[3][0][1]))
        self.cov_14.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[3][1][0]))
        self.cov_15.setText("{0:.4f}".format(icd.DEFAULT_FOURELLIPSE_COV[3][1][1]))

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_box)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox)
        testLayout.addWidget(self.horizontalGroupBox1)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox2)
        testLayout.addWidget(self.horizontalGroupBox3)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox4)
        testLayout.addWidget(self.horizontalGroupBox5)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox6)
        testLayout.addWidget(self.horizontalGroupBox7)
        self.windowLayout.addLayout(testLayout)

        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        # if train button selected, plot in train window                 
        #  note that the lambda function creates a "callable" object     
        #  which is required by connect()                                   
        #                               
        if sender == WINDOW_TRAIN:
            button.clicked.connect\
                (lambda:self.event_loop.plot_four_ellip(WINDOW_TRAIN))

        # otherwise, plot in eval window
        #
        elif sender == WINDOW_EVAL:
            button.clicked.connect\
                (lambda:self.event_loop.plot_four_ellip(WINDOW_EVAL))

        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    # method: Second::rotated_ellipses
    #
    # arguments:
    #  sender: title of button pushed that invoked this method
    #
    # This method creates the parameter window which displays after the
    #  "Rotated Ellipses" pattern is selected
    #
    def rotated_ellipses(self,sender):
        self.create_grid_layout_two()

        # create placeholder text for all text box
        #
        self.num_points.setText(str(icd.DEFAULT_NPTS_PER_CLASS))

        self.mu_0.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_MEAN[0][0]))
        self.mu_1.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_MEAN[0][1]))

        self.mu_2.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_MEAN[1][0]))
        self.mu_3.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_MEAN[1][1]))

        self.cov_0.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_COV[0][0][0]))
        self.cov_1.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_COV[0][0][1]))
        self.cov_2.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_COV[0][1][0]))
        self.cov_3.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_COV[0][1][1]))

        self.cov_4.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_COV[1][0][0]))
        self.cov_5.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_COV[1][0][1]))
        self.cov_6.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_COV[1][1][0]))
        self.cov_7.setText("{0:.4f}".format(icd.DEFAULT_ROTATEDELLIPSE_COV[1][1][1]))

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_box)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox)
        testLayout.addWidget(self.horizontalGroupBox1)
        self.windowLayout.addLayout(testLayout)
        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox2)
        testLayout.addWidget(self.horizontalGroupBox3)
        self.windowLayout.addLayout(testLayout)

        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        # if train button selected, plot in train window     
        #  note that the lambda function creates a "callable" object 
        #  which is required by connect()                           
        #                               
        if sender == WINDOW_TRAIN:
            button.clicked.connect\
                (lambda:self.event_loop.plot_rotated_ellips(WINDOW_TRAIN))

        # otherwise plot in eval window
        #
        elif sender == WINDOW_EVAL:
            button.clicked.connect\
                (lambda:self.event_loop.plot_rotated_ellips(WINDOW_EVAL))

        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    # method: Second::toroidal
    #
    # arguments:
    #  sender: title of button pushed that invoked this method
    #
    # This method creates the parameter window which displays after the
    #  "Toroidal" pattern is selected by the user
    #
    def toroidal(self,sender):
        self.create_grid_layout_torodial()

        # create plaecholder text for all text box
        #
        self.num_points.setText(str(icd.DEFAULT_NPTS_PER_CLASS))
        self.num_points_tor.setText("{0:.4f}".format(icd.DEFAULT_TOROID_NPTS_RING))

        self.inner_ring.setText("{0:.4f}".format(icd.DEFAULT_TOROID_INNER_RADIUS))
        self.outer_ring.setText("{0:.4f}".format(icd.DEFAULT_TOROID_OUTER_RADIUS))

        self.mu_0.setText("{0:.4f}".format(icd.DEFAULT_TOROID_INNER_MEAN[0]))
        self.mu_1.setText("{0:.4f}".format(icd.DEFAULT_TOROID_INNER_MEAN[1]))

        self.cov_0.setText("{0:.4f}".format(icd.DEFAULT_TOROID_INNER_COV[0][0]))
        self.cov_1.setText("{0:.4f}".format(icd.DEFAULT_TOROID_INNER_COV[0][1]))
        self.cov_2.setText("{0:.4f}".format(icd.DEFAULT_TOROID_INNER_COV[1][0]))
        self.cov_3.setText("{0:.4f}".format(icd.DEFAULT_TOROID_INNER_COV[1][1]))


        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_box_t)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.inner_and_outer_radius)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.horizontalGroupBox)
        testLayout.addWidget(self.horizontalGroupBox1)
        self.windowLayout.addLayout(testLayout)

        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        # if train button selected, plot in train window     
        #  note that the lambda function creates a "callable" object 
        #  which is required by connect()                           
        #
        if sender == WINDOW_TRAIN:
            button.clicked.connect\
                (lambda:self.event_loop.plot_toroidal(WINDOW_TRAIN))

        # otherwise plot in eval window
        #
        elif sender == WINDOW_EVAL:
            button.clicked.connect\
                (lambda:self.event_loop.plot_toroidal(WINDOW_EVAL))

        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    # method: Second::yin_yang
    #
    # arguments:
    #  sender: title of button pushed that invoked method
    #
    # This method creates the layout for the Yin-Yang pattern parameter
    # window.
    #
    def yin_yang(self, sender):

        # create grid layout for yin-yang parameters
        #
        self.create_grid_layout_yin_yang()

        # create placeholder text for parameters
        #
        self.num_points_yin.setText(str(icd.DEFAULT_YING_NPTS))
        self.num_points_yang.setText(str(icd.DEFAULT_YANG_NPTS))

        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_yin_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.num_points_yang_box)
        self.windowLayout.addLayout(testLayout)
        

        # add "Submit" button to params window
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        # if train button selected, plot in train window     
        #  note that the lambda function creates a "callable" object 
        #  which is required by connect()                           
        #        
        if sender == WINDOW_TRAIN:
            button.clicked.connect\
                (lambda:self.event_loop.plot_yin_yang(WINDOW_TRAIN))

        # otherwise plot in eval window
        #
        elif sender == WINDOW_EVAL:
            button.clicked.connect\
                (lambda:self.event_loop.plot_yin_yang(WINDOW_EVAL))

        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    # method: Second::create_grid_layout_two
    #
    # This method creates the layout of the parameter window which pops up
    #  when a user selects either the "Two Gaussian" or "Two Ellipses"
    #  from the menu.
    #
    def create_grid_layout_one(self):
        self.layout.addWidget(self.mu_0, 0, 0)
        self.layout.addWidget(self.mu_1, 1, 0)

        self.layout1.addWidget(self.cov_0, 0, 0)
        self.layout1.addWidget(self.cov_1, 0, 1)
        self.layout1.addWidget(self.cov_2, 1, 0)
        self.layout1.addWidget(self.cov_3, 1, 1)

        self.layout4.addWidget(self.num_size, 0, 0)

        self.size_of_gauss_box.setLayout(self.layout4)

        self.horizontalGroupBox.setLayout(self.layout)
        self.horizontalGroupBox1.setLayout(self.layout1)
        
    # method: Second::create_grid_layout_two
    #
    # This method creates the layout of the parameter window which pops up
    #  when a user selects either the "Two Gaussian" or "Two Ellipses"
    #  from the menu.
    #
    def create_grid_layout_two(self):
        self.layout.addWidget(self.mu_0, 0, 0)
        self.layout.addWidget(self.mu_1, 1, 0)

        self.layout1.addWidget(self.cov_0, 0, 0)
        self.layout1.addWidget(self.cov_1, 0, 1)
        self.layout1.addWidget(self.cov_2, 1, 0)
        self.layout1.addWidget(self.cov_3, 1, 1)


        self.layout2.addWidget(self.mu_2, 0, 0)
        self.layout2.addWidget(self.mu_3, 1, 0)

        self.layout3.addWidget(self.cov_4, 0, 0)
        self.layout3.addWidget(self.cov_5, 0, 1)
        self.layout3.addWidget(self.cov_6, 1, 0)
        self.layout3.addWidget(self.cov_7, 1, 1)

        self.layout4.addWidget(self.num_points, 0, 0)

        self.num_points_box.setLayout(self.layout4)

        self.horizontalGroupBox.setLayout(self.layout)
        self.horizontalGroupBox1.setLayout(self.layout1)

        self.horizontalGroupBox2.setLayout(self.layout2)
        self.horizontalGroupBox3.setLayout(self.layout3)

    def create_grid_layout_yin_yang(self):
        self.layout.addWidget(self.num_points_yin, 0, 0)
        self.layout1.addWidget(self.num_points_yang, 0, 0)

        self.num_points_yin_box.setLayout(self.layout)
        self.num_points_yang_box.setLayout(self.layout1)   
        
    # method: Second::create_grid_layout_four                                 
    #                                                                         
    # This method creates the layout of the parameter window which pops up    
    #  when a user selects either the "Four Gaussian" or "Four Ellipses"   
    #  patterns from the menu.
    #
    def create_grid_layout_four(self):
        self.layout.addWidget(self.mu_0, 0, 0)
        self.layout.addWidget(self.mu_1, 1, 0)

        self.layout1.addWidget(self.cov_0, 0, 0)
        self.layout1.addWidget(self.cov_1, 0, 1)
        self.layout1.addWidget(self.cov_2, 1, 0)
        self.layout1.addWidget(self.cov_3, 1, 1)


        self.layout2.addWidget(self.mu_2, 0, 0)
        self.layout2.addWidget(self.mu_3, 1, 0)

        self.layout3.addWidget(self.cov_4, 0, 0)
        self.layout3.addWidget(self.cov_5, 0, 1)
        self.layout3.addWidget(self.cov_6, 1, 0)
        self.layout3.addWidget(self.cov_7, 1, 1)

        self.layout4.addWidget(self.num_points, 0, 0)

        self.layout5.addWidget(self.mu_4, 0, 0)
        self.layout5.addWidget(self.mu_5, 1, 0)

        self.layout6.addWidget(self.cov_8, 0, 0)
        self.layout6.addWidget(self.cov_9, 0, 1)
        self.layout6.addWidget(self.cov_10, 1, 0)
        self.layout6.addWidget(self.cov_11, 1, 1)

        self.layout7.addWidget(self.mu_6, 0, 0)
        self.layout7.addWidget(self.mu_7, 1, 0)

        self.layout8.addWidget(self.cov_12, 0, 0)
        self.layout8.addWidget(self.cov_13, 0, 1)
        self.layout8.addWidget(self.cov_14, 1, 0)
        self.layout8.addWidget(self.cov_15, 1, 1)

        self.num_points_box.setLayout(self.layout4)

        self.horizontalGroupBox.setLayout(self.layout)
        self.horizontalGroupBox1.setLayout(self.layout1)

        self.horizontalGroupBox2.setLayout(self.layout2)
        self.horizontalGroupBox3.setLayout(self.layout3)

        self.horizontalGroupBox4.setLayout(self.layout5)
        self.horizontalGroupBox5.setLayout(self.layout6)

        self.horizontalGroupBox6.setLayout(self.layout7)
        self.horizontalGroupBox7.setLayout(self.layout8)
        
    # method: Second::create_grid_layout_torodial                              
    #                                                                         
    # This method creates the layout of the parameter window which pops up    
    #  when a user selects the "Torodial" pattern from the menu.        
    #
    def create_grid_layout_torodial(self):
        self.layout.addWidget(self.num_points, 0, 0)
        self.layout1.addWidget(self.num_points_tor, 0, 0)

        self.layout2.addWidget(self.mu_0, 0, 0)
        self.layout2.addWidget(self.mu_1, 1, 0)

        self.layout3.addWidget(self.cov_0, 0, 0)
        self.layout3.addWidget(self.cov_1, 0, 1)
        self.layout3.addWidget(self.cov_2, 1, 0)
        self.layout3.addWidget(self.cov_3, 1, 1)

        self.layout4.addWidget(self.inner_ring, 0, 0)
        self.layout4.addWidget(self.outer_ring, 0, 1)

        self.num_points_box.setLayout(self.layout)
        self.num_points_box_t.setLayout(self.layout1)

        self.horizontalGroupBox.setLayout(self.layout2)
        self.horizontalGroupBox1.setLayout(self.layout3)

        self.inner_and_outer_radius.setLayout(self.layout4)

    def knn_params(self, sender):
        
        # layout 
        #
        self.layout.addWidget(self.knn_neighbor_line, 0, 0)
        self.knn_neighbor_box.setLayout(self.layout)

        self.knn_weight_selection = QtWidgets.QComboBox()
        self.knn_weight_selection.addItems(KNN_WEIGHT)
        self.layout1.addWidget(self.knn_weight_selection, 0, 0)
        self.knn_weights_box.setLayout(self.layout1)

        self.knn_algo_selection = QtWidgets.QComboBox()
        self.knn_algo_selection.addItems(KNN_ALGO)
        self.layout2.addWidget(self.knn_algo_selection, 0, 0)
        self.knn_algo_box.setLayout(self.layout2)

        # create text for all text box
        #
        self.knn_neighbor_line.setText("5")

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.knn_neighbor_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.knn_weights_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.knn_algo_box)
        self.windowLayout.addLayout(testLayout)

        # set up button for submission
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        button.clicked.connect\
                (lambda:self.event_loop.algorithm_knn())
        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)  

    def svm_params(self, sender):

        # layout 
        #
        self.layout.addWidget(self.svm_max_iter_line, 0, 0)
        self.svm_max_iter_box.setLayout(self.layout)
        self.svm_max_iter_line.setText("1")

        # layout 
        #
        self.svm_gamma_selection = QtWidgets.QComboBox()
        self.svm_gamma_selection.addItems(SVM_GAMMA)
        self.layout1.addWidget(self.svm_gamma_selection, 0, 0)
        self.svm_gamma_box.setLayout(self.layout1)
        
        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.svm_max_iter_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.svm_gamma_box)
        self.windowLayout.addLayout(testLayout)

        # set up button for submission
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        button.clicked.connect\
                (lambda:self.event_loop.algorithm_svm())
        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)  

    def kmeans_params(self, sender):

        # layout 
        #
        self.layout.addWidget(self.kmeans_ncluster_line, 0, 0)
        self.kmeans_ncluster_box.setLayout(self.layout)

        self.kmeans_init_selection = QtWidgets.QComboBox()
        self.kmeans_init_selection.addItems(KMEANS_INIT)
        self.layout1.addWidget(self.kmeans_init_selection, 0, 0)
        self.kmeans_init_box.setLayout(self.layout1)

        self.layout2.addWidget(self.kmeans_ninit_line, 0, 0)
        self.kmeans_ninit_box.setLayout(self.layout2)

        self.layout3.addWidget(self.kmeans_max_iter_line, 0, 0)
        self.kmeans_max_iter_box.setLayout(self.layout3)
        
        # create text for all text box
        #
        self.kmeans_ncluster_line.setText("8")
        self.kmeans_ninit_line.setText("10")
        self.kmeans_max_iter_line.setText("300")
        
        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.kmeans_ncluster_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.kmeans_init_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.kmeans_ninit_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.kmeans_max_iter_box)
        self.windowLayout.addLayout(testLayout)

        # set up button for submission
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        button.clicked.connect\
                (lambda:self.event_loop.algorithm_kmeans())
        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    def mlp_params(self, sender):

        # layout 
        #
        self.layout.addWidget(self.mlp_layer_line, 0, 0)
        self.mlp_layer_box.setLayout(self.layout)

        self.mlp_solver_selection = QtWidgets.QComboBox()
        self.mlp_solver_selection.addItems(MLP_SOLVER)
        self.layout1.addWidget(self.mlp_solver_selection, 0, 0)
        self.mlp_solver_box.setLayout(self.layout1)

        # create text for all text box
        #
        self.mlp_layer_line.setText("(100,)")

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.mlp_layer_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.mlp_solver_box)
        self.windowLayout.addLayout(testLayout)

        # set up button for submission
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        button.clicked.connect\
                (lambda:self.event_loop.algorithm_mlp())
        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

    def rf_params(self, sender):

        # layout 
        #
        self.layout.addWidget(self.rf_ntree_line, 0, 0)
        self.rf_ntree_box.setLayout(self.layout)
        self.rf_crit_selection = QtWidgets.QComboBox()
        self.rf_crit_selection.addItems(RF_CRIT)
        self.layout1.addWidget(self.rf_crit_selection, 0, 0)
        self.rf_crit_box.setLayout(self.layout1)

        # create text for all text box
        #
        self.rf_ntree_line.setText("100")

        # creates overall layout for the window
        #
        self.windowLayout = QtWidgets.QVBoxLayout()

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.rf_ntree_box)
        self.windowLayout.addLayout(testLayout)

        testLayout = QtWidgets.QHBoxLayout()
        testLayout.addWidget(self.rf_crit_box)
        self.windowLayout.addLayout(testLayout)

        # set up button for submission
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        button.clicked.connect\
                (lambda:self.event_loop.algorithm_rf())
        button.clicked.connect(lambda: self.close())
        self.setLayout(self.windowLayout)

# class: Settings
#
# arguments:
#  QtWidgets.QDialog: Qt base class of dialog widgets
#
# This class implements the Settings option in the menu bar for the ISIP
# Machine Learning Demo user interface. These settings affect the
# "Change Range" option in the Menu Bar.
#
class Settings(QtWidgets.QDialog):

    # method: Settings::constructor
    #
    # arguments:
    #  event_loop: the event loop
    #
    def __init__(self, event_loop):
        super().__init__()
        self.event_loop = event_loop

        self.title = "Set Ranges"
        self.width = 240
        self.height = 100

        # layouts to be used for Range Setting
        #
        self.train_group_box = QtWidgets.QGroupBox("Train")
        self.eval_group_box = QtWidgets.QGroupBox("Eval")

        # layouts for input
        #
        self.layout_0 = QtWidgets.QGridLayout()
        self.layout_1 = QtWidgets.QGridLayout()

        # Input Boxes to be used for settings
        #
        self.input_0 = QtWidgets.QLineEdit()
        self.input_1 = QtWidgets.QLineEdit()
        self.input_2 = QtWidgets.QLineEdit()
        self.input_3 = QtWidgets.QLineEdit()
        self.input_4 = QtWidgets.QLineEdit()
        self.input_5 = QtWidgets.QLineEdit()
        self.input_6 = QtWidgets.QLineEdit()
        self.input_7 = QtWidgets.QLineEdit()

        self.initUI()

    # method: Settings::initUI
    #
    # This method initializes the visuals for settings
    #
    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(self.width, self.height)

        scriptDir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep + 'logo.png'))

    # method: Settings::plot_ranges
    #
    # This method brings up a secondary window to prompt user for new
    # ranges for the X and Y axes of the Train and Eval windows.
    #
    def plot_ranges(self):

        # set up the seconod window for ranges
        #
        self.layout_0.addWidget(self.input_0, 0, 0)
        self.layout_0.addWidget(self.input_1, 0, 1)
        self.layout_0.addWidget(self.input_2, 1, 0)
        self.layout_0.addWidget(self.input_3, 1, 1)

        self.train_group_box.setLayout(self.layout_0)

        self.layout_1.addWidget(self.input_4, 0, 0)
        self.layout_1.addWidget(self.input_5, 0, 1)
        self.layout_1.addWidget(self.input_6, 1, 0)
        self.layout_1.addWidget(self.input_7, 1, 1)

        self.eval_group_box.setLayout(self.layout_1)

        # set placehokder text representing what the values should be
        #
        self.input_0.setText("X min")
        self.input_1.setText("X max")

        self.input_2.setText("Y mix")
        self.input_3.setText("Y max")

        self.input_4.setText("X min")
        self.input_5.setText("X max")

        self.input_6.setText("Y mix")
        self.input_7.setText("Y max")

        self.windowLayout = QtWidgets.QVBoxLayout()

        current_layout = QtWidgets.QHBoxLayout()
        current_layout.addWidget(self.train_group_box)
        self.windowLayout.addLayout(current_layout)

        current_layout = QtWidgets.QHBoxLayout()
        current_layout.addWidget(self.eval_group_box)
        self.windowLayout.addLayout(current_layout)

        # create the submit button and link it back to the event loop
        #
        button = QtWidgets.QPushButton('Submit', self)
        self.windowLayout.addWidget(button)

        button.clicked.connect(self.event_loop.change_range)

        self.setLayout(self.windowLayout)

#
# end of class

#
# end of file
