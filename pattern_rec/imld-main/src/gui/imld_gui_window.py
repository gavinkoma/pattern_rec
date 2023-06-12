# file: imld_window.py
#
# This script implements the main window, menu bar and 3 widgets in the ISIP
# Machine Learning Demo user interface
# ------------------------------------------------------------------------------

# import system modules
#
import numpy as np
import os
import time

# import other modules
#
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg \
    as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import cnames
from matplotlib.pyplot import colormaps
from PyQt5 import QtCore, QtGui, QtWidgets
import alg.imld_model as model

# ------------------------------------------------------------------------------
#                                                                           
# define global variables                                                  
#                                                                          
# ------------------------------------------------------------------------------

# define algorithm names
#
ALG_PCA_CI = "Class Independent Principle Component Analysis (CI-PCA)"
ALG_PCA_CD = "Class Dependent Principle Component Analysis (CD-PCA)"
ALG_LDA_CI = "Class Independent LDA (CI-LDA)"
ALG_LDA_CD = "Class Dependent LDA (CD-LDA)"
ALG_NEAREST = "K Nearest Neighbor (KNN)"
ALG_KMEANS = "K-Means (KM)"
ALG_LBG = "Linde-Buzo-Gray (LBG)"
ALG_SUPPORT = "Support Vector Machines (SVM)"
ALG_MLP = "Multilayer Perceptron (MLP)"
ALG_RF = "Random Forest (RF)"

# define window parameter
#
INPUT_DISP_ROW = 0
INPUT_DISP_COL = 0
OUTPUT_DISP_ROW = 2
OUTPUT_DISP_COL = 0
PROCES_ROW = 0
PROCES_COL = 1
PROCES_ROW_SPAN = 0
PROCES_COL_SPAN = 2
WINDOW_X = 800
WINDOW_Y = 768
SURFACE_COLOR = 'winter'
COLORS = [*cnames]
DEFAULT = 1


# class: MainWindow
#
# This class uses QMainWindow which provides a main application window. The
#  modules in this class handles the graphics of the UI and logic that connects
#  the UI to specific actions.
#
class MainWindow(QtWidgets.QMainWindow):

    # method: MainWindow::constructor
    #
    # arguments:
    #  event_loop_a: the event loop
    #
    def __init__(self, event_loop_a):
        super(MainWindow, self).__init__()

        # add all classes from QMainWindow so they may be referenced
        # by self.class
        #
        QtWidgets.QMainWindow.__init__(self)

        # store reference to parent loop
        #
        self.parent_loop = event_loop_a

        # create instance  of widgets including the train display, eval
        # display and progress description
        #
        self.input_display = InputDisplay()
        self.output_display = OutputDisplay()
        self.process_desc = ProcessDescription()

        # create instance of menu bar at top of window using MenuBar
        #
        self.menu_bar = MenuBar()

        # create layout for applet, set style of frame and lay out
        # widgets in a grid
        #
        self.central_widget = QtWidgets.QFrame()
        self.central_widget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.layout = QtWidgets.QGridLayout(self.central_widget)
        self.central_widget.setStyleSheet("background-color: white;")

        # initialize window
        #
        self.initUI()

    # method: MainWindow::initUI
    #
    # This method lays out and positions widgets, creates a menu bar and
    #  connects the physical menu created in class MenuBar to the actions
    #  required when user selects an option
    #
    def initUI(self):
        # give the main window a title
        #
        self.setWindowTitle("ISIP Machine Learning Demonstration")

        # set app icon    
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        app_icon.addFile(scriptDir + os.path.sep + 'logo_16.png', \
                         QtCore.QSize(16, 16))
        app_icon.addFile(scriptDir + os.path.sep + 'logo_24.png', \
                         QtCore.QSize(24, 24))
        app_icon.addFile(scriptDir + os.path.sep + 'logo_32.png', \
                         QtCore.QSize(32, 32))
        app_icon.addFile(scriptDir + os.path.sep + 'logo_48.png', \
                         QtCore.QSize(48, 48))
        app_icon.addFile(scriptDir + os.path.sep + 'logo_256.png', \
                         QtCore.QSize(256, 256))

        self.setWindowIcon(QtGui.QIcon(scriptDir + os.path.sep + 'logo.png'))
        self.setWindowIcon(app_icon)

        # layout the widgets' positions in window:
        #
        self.setCentralWidget(self.central_widget)
        self.layout.addWidget(self.input_display, INPUT_DISP_ROW, INPUT_DISP_COL)
        self.layout.addWidget(self.output_display, OUTPUT_DISP_ROW, OUTPUT_DISP_COL)
        #self.layout.addWidget()
        self.layout.addWidget(self.process_desc, PROCES_ROW, PROCES_COL, PROCES_ROW_SPAN, PROCES_COL_SPAN)

        # make sure that the columns are shared equally
        #
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)

        # creates menu bar
        #
        self.setMenuBar(self.menu_bar)
        self.menu_bar.setNativeMenuBar(False)

        # ---------------------------------------------------------------------
        # connect pattern bar actions to event loop
        # ---------------------------------------------------------------------

        # IMLD menu
        #
        self.menu_bar.quit.triggered.connect(
            QtWidgets.QApplication.quit)

        # create instance of text edit for IMLD > About menu
        #
        self.menu_bar.about_menu.triggered.connect(
            self.parent_loop.about_IMLD)

        # file menu (load and save)
        #
        self.menu_bar.load_train_menu.triggered.connect(
            self.parent_loop.prompt_for_load_train)
        self.menu_bar.load_eval_menu.triggered.connect(
            self.parent_loop.prompt_for_load_eval)

        self.menu_bar.save_train_menu.triggered.connect(
            self.parent_loop.prompt_for_save_train)
        self.menu_bar.save_eval_menu.triggered.connect(
            self.parent_loop.prompt_for_save_eval)

        # when Edit > Set Range menu option selected, open set plot
        # ranges window
        #
        self.menu_bar.set_range_menu.triggered.connect(
            self.parent_loop.set_plot_ranges)
        self.menu_bar.set_scolor_menu.triggered.connect(
            self.parent_loop.surface_color_show)

        # when Edit > Set Gaussian menu option selected, open gaussian
        # pattern window
        #
        self.menu_bar.set_gaussian_menu.triggered.connect(
            self.parent_loop.prompt_set_gauss_prop)
        
        # when Edit > Normalize Data menu option selected, call normalize
        # data
        #
        self.menu_bar.set_normalize_menu.triggered.connect(
            self.parent_loop.normalize_data)

        # when Edit > Clear Descriptions selected, clear the window
        #
        self.menu_bar.clear_des_menu.triggered.connect(
            self.process_desc.output.clear)

        self.menu_bar.clear_des_menu.triggered.connect(
            self.process_desc.clear_progress)

        # when Edit > Clear Train
        #
        self.menu_bar.clear_train_data.triggered.connect(
            lambda checked: self.parent_loop.clear_input(checked))

        self.menu_bar.clear_train_result.triggered.connect(
            lambda checked: self.parent_loop.clear_input_result(checked))

        self.menu_bar.clear_train_all.triggered.connect(
            lambda checked: self.parent_loop.clear_input(checked))

        # when Edit > Clear Eval
        #
        self.menu_bar.clear_eval_data.triggered.connect(
            lambda checked: self.parent_loop.clear_output(checked))

        self.menu_bar.clear_eval_result.triggered.connect(
            lambda checked: self.parent_loop.clear_output_result(checked))

        self.menu_bar.clear_eval_all.triggered.connect(
            lambda checked: self.parent_loop.clear_output(checked))

        # when Edit > Clear All, clear Train, Eval, and Process Log
        #
        self.menu_bar.clear_all_menu.triggered.connect(
            lambda checked: self.parent_loop.clear_input(checked))

        self.menu_bar.clear_all_menu.triggered.connect(
            lambda checked: self.parent_loop.clear_output(checked))

        self.menu_bar.clear_all_menu.triggered.connect(
            self.process_desc.output.clear)

        self.menu_bar.reset_menu.triggered.connect(
            lambda checked: self.parent_loop.reset_window(checked))

        # when Classes > Add Class, Delete Class
        #
        self.menu_bar.add_class_menu.triggered.connect(
            self.parent_loop.add_class_show)

        # resets the color chosen when adding classes
        #
        self.menu_bar.add_class_menu.triggered.connect(
            self.parent_loop.reset_color)

        # handles the signal of the added class
        #
        self.menu_bar.class_group.triggered.connect(
            (lambda checked: self.parent_loop.handled_signal(
                checked, self.menu_bar.class_group.sender())))

        # removes the class selected
        #
        self.menu_bar.delete_class_menu.triggered.connect(
            lambda checked: self.parent_loop.remove_classes(checked))

        # when Algorithms > PCA, LDA, SVM, KNN, KMean, MLP
        # when LDA > Independent (ID), Dependent (CD)
        #
        self.menu_bar.algo_lda_cd_menu.triggered.connect(
            self.parent_loop.algorithm_lda_cd)

        self.menu_bar.algo_lda_ci_menu.triggered.connect(
            self.parent_loop.algorithm_lda_ci)

        # when PCA > Independent (ID), Dependent (CD)
        #
        self.menu_bar.algo_pca_cd_menu.triggered.connect(
            self.parent_loop.algorithm_pca_cd)

        self.menu_bar.algo_pca_ci_menu.triggered.connect(
            self.parent_loop.algorithm_pca_ci)

        # When Algorithms > SVM, KNN, KMean, MLP
        #
        self.menu_bar.algo_svm_menu.triggered.connect(
            self.parent_loop.prompt_algo_svm)

        self.menu_bar.algo_knn_menu.triggered.connect(
            self.parent_loop.prompt_algo_knn)

        self.menu_bar.algo_kmean_menu.triggered.connect(
            self.parent_loop.prompt_algo_kmeans)

        self.menu_bar.algo_mlp_menu.triggered.connect(
            self.parent_loop.prompt_algo_mlp)
        self.menu_bar.algo_rf_menu.triggered.connect(
            self.parent_loop.prompt_algo_rf)

        # patterns menu:
        #  These will bring up the parameter windows from the _show methods
        #  in event_loop when the menu option is selected.
        #

        # handles connecting GUI to draw point and gaussian
        #
        self.menu_bar.draw_points_menu.triggered.connect(
            self.parent_loop.set_point)

        # self.menu_bar.draw_points_menu.triggered.connect(
        #     self.menu_bar.draw_gauss_menu.setChecked(False))

        self.menu_bar.draw_gauss_menu.triggered.connect(
            self.parent_loop.set_gauss)

        # handles connecting GUI to showing two gaussian
        #
        self.menu_bar.two_gauss_menu_t.triggered.connect(
            self.parent_loop.two_gauss_show)
        self.menu_bar.two_gauss_menu_e.triggered.connect(
            self.parent_loop.two_gauss_show)

        # handles connecting GUI to showing four gaussian
        #
        self.menu_bar.four_gauss_menu_t.triggered.connect(
            self.parent_loop.four_gauss_show)
        self.menu_bar.four_gauss_menu_e.triggered.connect(
            self.parent_loop.four_gauss_show)

        # handles connecting GUI to showing over gaussian
        #
        self.menu_bar.over_gauss_menu_t.triggered.connect(
            self.parent_loop.over_gauss_show)
        self.menu_bar.over_gauss_menu_e.triggered.connect(
            self.parent_loop.over_gauss_show)

        # handles connecting GUI to showing two ellipse
        #
        self.menu_bar.two_ellipse_menu_t.triggered.connect(
            self.parent_loop.two_ellipse_show)
        self.menu_bar.two_ellipse_menu_e.triggered.connect(
            self.parent_loop.two_ellipse_show)

        # handles connecting GUI to showing four ellipse
        #
        self.menu_bar.four_ellipse_menu_t.triggered.connect(
            self.parent_loop.four_ellipse_show)
        self.menu_bar.four_ellipse_menu_e.triggered.connect(
            self.parent_loop.four_ellipse_show)

        # handles connecting GUI to showing rotated ellipse
        #
        self.menu_bar.rotated_ellipse_menu_t.triggered.connect(
            self.parent_loop.rotated_ellipse_show)
        self.menu_bar.rotated_ellipse_menu_e.triggered.connect(
            self.parent_loop.rotated_ellipse_show)

        # handles connecting GUI to showing toroidal
        #
        self.menu_bar.toroidal_menu_t.triggered.connect(
            self.parent_loop.toroidal_show)

        self.menu_bar.toroidal_menu_e.triggered.connect(
            self.parent_loop.toroidal_show)

        # handles connecting GUI to showing yin_yang symbol
        #
        self.menu_bar.yin_yang_menu_t.triggered.connect(
            self.parent_loop.yin_yang_show)

        self.menu_bar.yin_yang_menu_e.triggered.connect(
            self.parent_loop.yin_yang_show)

        # run menu
        #
#        self.menu_bar.run_menu.triggered.connect(self.process_desc.start_progress)
        self.menu_bar.run_menu.triggered.connect(
            self.parent_loop.run_complete)

        self.menu_bar.step_menu.triggered.connect(
            self.parent_loop.run_step)

        #self.menu_bar.step_menu.triggered.connect(self.process_desc.start_progress)

        # resize to specified dimension
        #
        self.resize(WINDOW_X, WINDOW_Y)

        # finally, show the window
        #
        self.show()


# class: MenuBar
#
# arguments:
#  QMenuBar: QT class which provides a horizontal menu bar
#
# This class physically adds options, suboptions and other formatting items
# to the menu bar.
#
class MenuBar(QtWidgets.QMenuBar):

    # method: MenuBar::constructor
    #
    def __init__(self):
        # inherit methods of PyQt object QMenuBar
        #
        QtWidgets.QMenuBar.__init__(self)

        # create menu bar options
        #
        self.imld_menu = self.addMenu('IMLD')
        self.file_menu = self.addMenu('File')
        self.edit_menu = self.addMenu('Edit')
        self.class_menu = self.addMenu('Classes')
        self.pattern_menu = self.addMenu('Patterns')
        self.demo_menu = self.addMenu('Demo')
        self.algo_menu = self.addMenu('Algorithms')
        self.process_menu = self.addMenu('Process')

        # IMLD menu bar
        # create options
        #
        self.about_menu = QtWidgets.QAction('About', self)
        self.quit = QtWidgets.QAction('Quit', self)

        # add options to IMLD slide down menu
        #
        self.imld_menu.addAction(self.about_menu)
        self.imld_menu.addAction(self.quit)

        # File menu bar
        # creates options for file slide down (load train)
        #
        self.load_train_menu = QtWidgets.QAction('Load Train Data', self)
        self.load_eval_menu = QtWidgets.QAction('Load Eval Data', self)

        # adds options to file slide down menu as well as a separator to
        # physically separate these from the next options
        #
        self.file_menu.addAction(self.load_train_menu)
        self.file_menu.addAction(self.load_eval_menu)
        self.file_menu.addSeparator()

        # creates options to save train/eval data
        #
        self.save_train_menu = QtWidgets.QAction('Save Train As...', self)
        self.save_eval_menu = QtWidgets.QAction('Save Eval As...', self)

        # adds save options to file menu
        #
        self.file_menu.addAction(self.save_train_menu)
        self.file_menu.addAction(self.save_eval_menu)

        # Edit menu bar

        # add 'Settings' sub menu to 'Edit' menu
        #
        self.settings_menu = self.edit_menu.addMenu("Settings")
        self.edit_menu.addSeparator()

        # create 'Set Ranges', 'Set Color', and 'Set Gaussian' options under 'Settings'
        # sub-menu
        #
        self.set_range_menu = QtWidgets.QAction('Set Ranges')
        self.set_gaussian_menu = QtWidgets.QAction('Set Gaussian')
        self.set_scolor_menu = QtWidgets.QAction('Set Color')
        self.set_normalize_menu = QtWidgets.QAction('Normalize Data', self, checkable = True, checked = False)

        # add options to 'Settings' menu
        #
        self.settings_menu.addAction(self.set_range_menu)
        self.settings_menu.addAction(self.set_gaussian_menu)
        self.settings_menu.addAction(self.set_scolor_menu)
        self.settings_menu.addAction(self.set_normalize_menu)
        
        # create 'Clear' options
        #
        self.clear_des_menu = QtWidgets.QAction('Clear Process Log')

        self.clear_all_menu = QtWidgets.QAction('Clear All')
        self.reset_menu = QtWidgets.QAction('Reset')

        self.clear_train_data = QtWidgets.QAction('Clear Data')
        self.clear_train_result = QtWidgets.QAction('Clear Results')
        self.clear_train_all = QtWidgets.QAction('Clear All')

        self.clear_eval_data = QtWidgets.QAction('Clear Data')
        self.clear_eval_result = QtWidgets.QAction('Clear Results')
        self.clear_eval_all = QtWidgets.QAction('Clear All')

        # add 'Clear' options under 'Edit' menu
        #
        self.edit_menu.addAction(self.clear_des_menu)

        # create Clear Train options
        #
        self.clear_train = self.edit_menu.addMenu('Clear Train')
        self.clear_train.addAction(self.clear_train_data)
        self.clear_train.addAction(self.clear_train_result)
        self.clear_train.addAction(self.clear_train_all)

        # create Clear Eval options
        #
        self.clear_eval = self.edit_menu.addMenu('Clear Eval')
        self.clear_eval.addAction(self.clear_eval_data)
        self.clear_eval.addAction(self.clear_eval_result)
        self.clear_eval.addAction(self.clear_eval_all)

        # create Clear All menu
        #
        self.edit_menu.addAction(self.clear_all_menu)
        self.edit_menu.addAction(self.reset_menu)

        # Classes menu bar

        # create an action group so that the checkable option can be
        # made exclusive
        #
        self.class_group = QtWidgets.QActionGroup(self.class_menu)

        # create sub-menu options for "Classes" menu
        #
        self.add_class_menu = QtWidgets.QAction("Add Class")
        self.delete_class_menu = QtWidgets.QAction("Delete Class")

        # add sub-menus Add Class and Delete Class, to "Classes" menu group
        #
        self.class_menu.addAction(self.add_class_menu)
        self.class_menu.addAction(self.delete_class_menu)
        self.class_menu.addSeparator()

        # Patterns menu bar
        # create options to add to pattern slide down
        # (Draw Options)
        #
        self.draw_points_menu = QtWidgets.QAction('Draw Points', self, checkable = True, checked = True)
        self.draw_gauss_menu = QtWidgets.QAction('Draw Gaussian', self, checkable = True)

        self.pattern_menu.addAction(self.draw_points_menu)
        self.pattern_menu.addAction(self.draw_gauss_menu)

        # Demo menu bar
        # create options for use of predefined demos
        # two gaussian - add options to create in Train or Eval window
        #
        self.two_gauss_menu = self.demo_menu.addMenu("Two Gaussian")
        self.two_gauss_menu_t = QtWidgets.QAction('Train')
        self.two_gauss_menu_e = QtWidgets.QAction('Eval')

        label = QtWidgets.QLabel("Set Parameters")
        label_action = QtWidgets.QWidgetAction(self)
        label_action.setDefaultWidget(label)

        self.two_gauss_menu.addAction(label_action)
        self.two_gauss_menu.addAction(self.two_gauss_menu_t)
        self.two_gauss_menu.addAction(self.two_gauss_menu_e)

        # four gaussian - add options to create in Train or Eval window
        #
        self.four_gauss_menu = self.demo_menu.addMenu("Four Gaussian")
        self.four_gauss_menu_t = QtWidgets.QAction('Train')
        self.four_gauss_menu_e = QtWidgets.QAction('Eval')

        label = QtWidgets.QLabel("Set Parameters")
        label_action = QtWidgets.QWidgetAction(self)
        label_action.setDefaultWidget(label)

        self.four_gauss_menu.addAction(label_action)
        self.four_gauss_menu.addAction(self.four_gauss_menu_t)
        self.four_gauss_menu.addAction(self.four_gauss_menu_e)

        # overlapping gaussian - add options to create in Train or Eval window
        #
        self.over_gauss_menu = \
            self.demo_menu.addMenu("Overlapping Gaussian")

        self.over_gauss_menu_t = QtWidgets.QAction('Train')
        self.over_gauss_menu_e = QtWidgets.QAction('Eval')

        label = QtWidgets.QLabel("Set Parameters")
        label_action = QtWidgets.QWidgetAction(self)
        label_action.setDefaultWidget(label)

        self.over_gauss_menu.addAction(label_action)
        self.over_gauss_menu.addAction(self.over_gauss_menu_t)
        self.over_gauss_menu.addAction(self.over_gauss_menu_e)

        # two ellipses - add options to create in Train or Eval window
        #
        self.two_ellipse_menu = self.demo_menu.addMenu("Two Ellipses")
        self.two_ellipse_menu_t = QtWidgets.QAction('Train')
        self.two_ellipse_menu_e = QtWidgets.QAction('Eval')

        label = QtWidgets.QLabel("Set Parameters")
        label_action = QtWidgets.QWidgetAction(self)
        label_action.setDefaultWidget(label)

        self.two_ellipse_menu.addAction(label_action)
        self.two_ellipse_menu.addAction(self.two_ellipse_menu_t)
        self.two_ellipse_menu.addAction(self.two_ellipse_menu_e)

        # four ellipses - add options to create in Train or Eval window
        #
        self.four_ellipse_menu = self.demo_menu.addMenu("Four Ellipses")
        self.four_ellipse_menu_t = QtWidgets.QAction('Train')
        self.four_ellipse_menu_e = QtWidgets.QAction('Eval')

        label = QtWidgets.QLabel("Set Parameters")
        label_action = QtWidgets.QWidgetAction(self)
        label_action.setDefaultWidget(label)

        self.four_ellipse_menu.addAction(label_action)
        self.four_ellipse_menu.addAction(self.four_ellipse_menu_t)
        self.four_ellipse_menu.addAction(self.four_ellipse_menu_e)

        # rotated ellipses - add options to create in Train or Eval window
        #
        self.rotated_ellipse_menu = \
            self.demo_menu.addMenu("Rotated Ellipses")

        self.rotated_ellipse_menu_t = QtWidgets.QAction('Train')
        self.rotated_ellipse_menu_e = QtWidgets.QAction('Eval')

        label = QtWidgets.QLabel("Set Parameters")
        label_action = QtWidgets.QWidgetAction(self)
        label_action.setDefaultWidget(label)

        self.rotated_ellipse_menu.addAction(label_action)
        self.rotated_ellipse_menu.addAction(self.rotated_ellipse_menu_t)
        self.rotated_ellipse_menu.addAction(self.rotated_ellipse_menu_e)

        # toroidal - add options to create in Train or Eval window
        #
        self.toroidal_menu = self.demo_menu.addMenu("Toroidal")
        self.toroidal_menu_t = QtWidgets.QAction('Train')
        self.toroidal_menu_e = QtWidgets.QAction('Eval')

        label = QtWidgets.QLabel("Set Parameters")
        label_action = QtWidgets.QWidgetAction(self)
        label_action.setDefaultWidget(label)

        self.toroidal_menu.addAction(label_action)
        self.toroidal_menu.addAction(self.toroidal_menu_t)
        self.toroidal_menu.addAction(self.toroidal_menu_e)

        # yin yang - add options to create in Train or Eval window
        #
        self.yin_yang_menu = self.demo_menu.addMenu("Yin-Yang")
        self.yin_yang_menu_t = QtWidgets.QAction('Train')
        self.yin_yang_menu_e = QtWidgets.QAction('Eval')

        label = QtWidgets.QLabel("Set Parameters")
        label_action = QtWidgets.QWidgetAction(self)
        label_action.setDefaultWidget(label)

        self.yin_yang_menu.addAction(label_action)
        self.yin_yang_menu.addAction(self.yin_yang_menu_t)
        self.yin_yang_menu.addAction(self.yin_yang_menu_e)

        # Algorithm Menu Bar
        # create group so that we can later make them checkable and exclusive
        # (only one can be checked at a time)
        #
        self.algo_group = QtWidgets.QActionGroup(self.algo_menu)

        # add menu options PCA to "Algorithm" menu choice
        #
        self.algo_pcas_menu = self.algo_menu.addMenu("PCA")

        # create sub-option for PCA sub-menu
        #
        self.algo_pca_ci_menu = QtWidgets.QAction(ALG_PCA_CI, checkable=True)
        self.algo_pca_cd_menu = QtWidgets.QAction(ALG_PCA_CD, checkable=True)

        # add sub-option to PCA sub-menu
        #
        self.algo_pcas_menu.addAction(self.algo_group.addAction
                                      (self.algo_pca_ci_menu))

        self.algo_pcas_menu.addAction(self.algo_group.addAction
                                      (self.algo_pca_cd_menu))

        # add menu option LDA to "Algorithm" menu choice
        #
        self.algo_ldas_menu = self.algo_menu.addMenu("LDA")

        # create sub-options for PCA menu option
        #
        self.algo_lda_ci_menu = QtWidgets.QAction(ALG_LDA_CI, checkable=True)
        self.algo_lda_cd_menu = QtWidgets.QAction(ALG_LDA_CD, checkable=True)

        # add sub-options to PCA menu
        #
        self.algo_ldas_menu.addAction(self.algo_group.addAction
                                      (self.algo_lda_ci_menu))
        self.algo_ldas_menu.addAction(self.algo_group.addAction
                                      (self.algo_lda_cd_menu))

        # create sub-menu options Nearest Neighbor, KMeans
        #
        self.algo_knn_menu = QtWidgets.QAction(ALG_NEAREST, checkable=True)
        self.algo_kmean_menu = QtWidgets.QAction(ALG_KMEANS, checkable=True)

        # add these sub-menus to "Algorithm" menu option
        #
        self.algo_menu.addAction(self.algo_group.addAction(self.algo_knn_menu))
        self.algo_menu.addAction(self.algo_group.addAction
                                 (self.algo_kmean_menu))

        # create sub-menu options Linde-Buzo-Gray, Support Vector Machines
        # and Multilayer Perceptron
        #
        self.algo_svm_menu = QtWidgets.QAction(ALG_SUPPORT, checkable=True)
        self.algo_mlp_menu = QtWidgets.QAction(ALG_MLP, checkable=True)
        self.algo_rf_menu = QtWidgets.QAction(ALG_RF, checkable=True)


        # add these sub-menus to the "Algorithm" menu option
        #
        self.algo_menu.addAction(self.algo_group.addAction(self.algo_svm_menu))
        self.algo_menu.addAction(self.algo_group.addAction(self.algo_mlp_menu))
        self.algo_menu.addAction(self.algo_group.addAction(self.algo_rf_menu))

        # Process menu bar
        # create sub-menu options to add to the "Process" menu option
        #
        self.step_menu = QtWidgets.QAction("Step")
        self.run_menu = QtWidgets.QAction("Run ...")

        # add sub-menus to "Process" menu
        #
        self.process_menu.addAction(self.step_menu)
        self.process_menu.addAction(self.run_menu)

        # set keyboard shortcuts for menu options
        #
        self.run_menu.setShortcut("Ctrl+R")
        self.step_menu.setShortcut("Ctrl+N")

        self.setStyleSheet("""
           QMenuBar {
               color: rgb(255,255,255);
               background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 lightgray, stop:1 darkgray);
               border-radius: 5px
               padding: 2px 10px;
               spacing: 3px;
           }
        """)


# class: InputDisplay
#
# arguments:
#  Qwidget: base class for all UI interface objects
#
# This class sets up the Train window (Input display).
# 
class InputDisplay(QtWidgets.QWidget):

    # method: constructor InputDisplay::constructor
    #
    # arguments:
    #  parent: parent widget (widget that drew child widget) set to None as
    #   default
    #
    def __init__(self, parent=None):
        super(InputDisplay, self).__init__(parent)

        # set up Canvas
        #
        self.canvas = FigureCanvas(Figure(tight_layout=True))
        self.label = QtWidgets.QLabel()

        # set up the x and y coordinate arrays
        #
        self.x = np.empty((0, 0))
        self.y = np.empty((0, 0))

        # set up the covariance of the gaussian
        # when user draws
        #
        self.cov = [[0.05, 0.0], [0.0, 0.05]]
        self.num_points = 25

        # set up color variables: color bank, colors used, and
        # current colors
        #
        self.colors = [*cnames]
        self.colors_used = []
        self.color_c = None
        self.surface_color = 'winter'

        # set up current class, triggered class, all_classes
        #
        self.current_class = None
        self.t_current_class = None
        self.all_classes = []
        self.current_co = None

        # set up dictionary to store class information
        #
        self.once_c = True
        self.pair = 0
        self.class_info = {}

        # set up initial press and draw flag
        #
        self.pressed = False
        self.draw = 'point'

        # Set title and maximum size in pixels for Train window
        #
        self.label.setText("Train:")
        self.label.setStyleSheet("color: black;")
        self.label.setMaximumSize(35, 10)

        # set up canvas layout
        #
        vertical_layout = QtWidgets.QVBoxLayout(self)
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        vertical_layout.addWidget(self.label)
        vertical_layout.addWidget(self.canvas)

        # set up canvas in GUI
        #
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.x_axis = [-DEFAULT, DEFAULT]
        self.y_axis = [-DEFAULT, DEFAULT]
        self.initUI()

        # set handles for pressing, releasing and moving
        #
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)

    # method: InputDisplay::initUI
    #
    # This method creates the input display graph overlay
    #
    def initUI(self):

        self.canvas.axes.set_xlim([self.x_axis[0], self.x_axis[-1]])
        self.canvas.axes.set_ylim([self.y_axis[0], self.y_axis[-1]])

        # set range and grid
        #
        self.canvas.axes.set_axisbelow(True)

        # set visual aspects of grid
        #
        self.canvas.axes.grid(which='major', color='grey', linestyle='-', linewidth=0.25, alpha=0.50)
        self.canvas.axes.tick_params(labelsize=8)

        self.x_axislim = self.canvas.axes.get_xlim()
        self.y_axislim = self.canvas.axes.get_ylim()
        self.x_axis = np.linspace(self.x_axislim[0], self.x_axislim[1], 9)
        self.y_axis = np.linspace(self.y_axislim[0], self.y_axislim[1], 9)
        self.canvas.axes.set_xticks(self.x_axis)
        self.canvas.axes.set_yticks(self.y_axis)

    # method: InputDisplay::clear_plot
    #
    # This method clears any data that is shown in the input display.
    #
    def clear_plot(self):

        # clear the input canvas and reset with the intial UI
        #
        self.canvas.axes.clear()
        self.initUI()

        # iterate through class info and reset all information while keeping the color of the class
        #
        count = 0
        for key in self.class_info:
            # reset x and y coordinates
            #
            self.class_info[key][1] = np.empty((0, 0))
            self.class_info[key][2] = np.empty((0, 0))

            # reset class plot with same color
            #
            self.class_info[key][0] = self.canvas.axes.scatter(self.class_info[key][1], self.class_info[key][2], s=1)
            self.class_info[key][0].set_color(self.class_info[key][4])

            # set current class to the key
            #
            self.current_class = key
            self.class_info[self.current_class][0].set_gid(
                np.full((1, np.shape(self.class_info[self.current_class][1])[0]),
                        count))

            self.canvas.draw_idle()
            count = count + 1

        self.canvas.draw_idle()

    # method: InputDisplay::clear_result_plot
    #
    # This method clears the results from the plot and keeps the data plotted
    #
    def clear_result_plot(self):

        # remove canvas and reinsert the initial UI
        #
        self.canvas.axes.clear()
        self.initUI()
        # iterate through class dictionary and replot the graphs without results
        #
        count = 0
        for key in self.class_info:
            self.class_info[key][0] = self.canvas.axes.scatter(self.class_info[key][1], self.class_info[key][2], s=1)
            self.class_info[key][0].set_color(self.class_info[key][4])
            self.current_class = key
            self.class_info[self.current_class][0].set_gid(
                np.full((1, np.shape(self.class_info[self.current_class][1])[0]),
                        count))
            count = count + 1
        self.canvas.draw_idle()

    # method: InputDisplay::remove_class
    #
    # arguments:
    # name: the current class that is chosen to be removed
    #
    # This method removes a single class using the delete class button.
    #
    def remove_class(self, name=None):
        if self.class_info:
            # clear the visual plot
            self.canvas.axes.clear()
            self.initUI()

            # set the current class and reset the colors used
            #
            tmp = self.current_class
            if self.class_info[self.current_class] is not None:
                color = self.class_info[self.current_class][4]
                if color in self.colors_used:
                    self.colors_used.remove(color)
                else:
                    self.colors.append(color)

                self.class_info.pop(self.current_class)

                # Replot the remaining classes from class_info with their appropriate colors
                #
                count = 0
                for key in self.class_info:
                    self.class_info[key][0] = self.canvas.axes.scatter(self.class_info[key][1], self.class_info[key][2],
                                                                       s=1)
                    self.class_info[key][0].set_color(self.class_info[key][4])
                    self.current_class = key
                    self.class_info[self.current_class][0].set_gid(
                        np.full((1, np.shape(self.class_info[self.current_class][1])[0]),
                                count))
                    self.canvas.draw_idle()
                    count = count + 1
                self.t_current_class.remove(tmp)
                self.canvas.draw_idle()

            else:
                pass

    # method: InputDisplay::on_press
    #
    # arguments:
    # event: the event loop for this app
    #
    # This method calculates where the mouse is on the display and records both the
    # the x and y coordinates of the data and stores them into a dictionary where
    # the class is linked with its data. There are two options either a click and drag
    # for points or use of gaussian plots. This makes up  the training data.
    #
    def on_press(self, event):
        
        # check if current class name is a string and if not make it a string
        #
        if not isinstance(self.current_class, str):
            self.current_class = "%s" % self.current_class

        # find the index of current class in total list
        #
        self.pair = self.t_current_class.index(self.current_class)

        # if the current class is not in the class dictionary or is initialized to Norn
        # initialize the class in the dictionary
        #
        if self.current_class not in self.class_info or (self.class_info[self.current_class] == None):
            self.class_info[self.current_class] = [self.current_c, self.x, self.y, self.once_c]

        # check to make sure mouse is within data input window
        #
        if isinstance(event.xdata, float) and isinstance(event.ydata, float):
            if self.x_axis[0] <= event.xdata <= self.x_axis[-1] \
                    and self.y_axis[0] <= event.ydata <= self.y_axis[-1]:
                # set the draw to point and track the x and y data
                if self.draw == "point":
                    self.pressed = True

                    if self.class_info[self.current_class][3]:
                        self.class_info[self.current_class][0] = self.canvas.axes.scatter(None, None, s=1)
                        self.class_info[self.current_class][3] = False
                    self.class_info[self.current_class][1] = np.append(self.class_info[self.current_class][1], event.xdata)
                    self.class_info[self.current_class][2] = np.append(self.class_info[self.current_class][2], event.ydata)
                    self.class_info[self.current_class][0].set_color(self.class_info[self.current_class][4])
                    self.canvas.draw_idle()

                # set the draw to gaussian and track the data produced by the target x,y and the
                # gaussian formula
                #
                else:
                    self.pressed = True
                    if self.class_info[self.current_class][3]:
                        self.class_info[self.current_class][0] = self.canvas.axes.scatter(None, None, s=1)
                        self.class_info[self.current_class][0].set_color(self.class_info[self.current_class][4])
                        self.class_info[self.current_class][3] = False

                    mu = [event.xdata, event.ydata]
                    cov = self.cov
                    gauss_plot = np.random.multivariate_normal(mu, cov, self.num_points)
                    self.class_info[self.current_class][1] = np.append(self.class_info[self.current_class][1], gauss_plot[:, 0])
                    self.class_info[self.current_class][2] = np.append(self.class_info[self.current_class][2], gauss_plot[:, 1])
                    self.class_info[self.current_class][0].set_offsets(
                        np.column_stack((self.class_info[self.current_class][1], self.class_info[self.current_class][2])))
                    self.canvas.draw_idle()

    # method: InputDisplay::on_move
    #
    # arguments:
    # event: the event loop for this app
    #
    # this method records the x and y data while the mouse is moving
    #
    def on_move(self, event):

        if self.pressed:
            # Check to make sure mouse is within data input window
            #
            if isinstance(event.xdata, float) and isinstance(event.ydata, float):
                if self.x_axis[0] <= event.xdata <= self.x_axis[-1] \
                        and self.y_axis[0] <= event.ydata <= self.y_axis[-1]:
                    # if set to point, append the x and y data of the current mouse position
                    #
                    if self.draw == "point":
                        self.class_info[self.current_class][1] = np.append(self.class_info[self.current_class][1], event.xdata)
                        self.class_info[self.current_class][2] = np.append(self.class_info[self.current_class][2], event.ydata)
                        self.class_info[self.current_class][0].set_offsets(
                            np.column_stack((self.class_info[self.current_class][1], self.class_info[self.current_class][2])))
                        self.canvas.draw_idle()

                    # if set to gaussian, record the gaussian with current mouse placement while moving
                    #
                    else:
                        mu = [event.xdata, event.ydata]
                        cov = self.cov
                        gauss_plot = np.random.multivariate_normal(mu, cov, self.num_points)
                        self.class_info[self.current_class][1] = np.append(self.class_info[self.current_class][1],
                                                                           gauss_plot[:, 0])
                        self.class_info[self.current_class][2] = np.append(self.class_info[self.current_class][2],
                                                                           gauss_plot[:, 1])
                        self.class_info[self.current_class][0].set_offsets(
                            np.column_stack((self.class_info[self.current_class][1], self.class_info[self.current_class][2])))
                        self.canvas.draw_idle()

    # method: InputDisplay::on_release
    #
    # arguments:
    # event: the event loop for this app
    #
    # reset the pressed flag and set a name for the class_info correlating with graphical data
    #
    def on_release(self, event):

        # set the pressed flag to false
        #
        self.pressed = False

        # set class in dictionary to pair with the graphical data
        #
        self.class_info[self.current_class][0].set_gid(np.full((1, np.shape(self.class_info[self.current_class][1])[0]),
                                                               self.pair))
        self.canvas.draw_idle()

    # method: InputDisplay::find_class_c
    #
    # arguments:
    #  sender: The menu item that is currently triggered
    #
    # This method finds the current class object that is triggered and finds what
    # the appropriate color that is paired with with class
    #
    def find_class_c(self, sender=None):

        # check if the sender signal is triggered
        #
        if sender is not None:
            # find the class pair of the signal and set it to the current class
            #
            self.pair = self.all_classes.index(sender)
            self.current_co = sender
            self.current_class = self.t_current_class[self.pair]

    # method: InputDisplay::set_point
    #
    # arguments: none
    #
    # set draw flag to point
    def set_point(self):
        self.draw = 'point'

    # method: InputDisplay::set_gauss
    #
    # arguments: none
    #
    # set draw flag to gauss
    def set_gauss(self):
        self.draw = 'gauss'

# class: OutputDisplay
#
# arguments:
#  Qwidget: base class for all UI interface objects
#
# This class sets up the Eval window (output display).
#
class OutputDisplay(QtWidgets.QWidget):

    # method: OutputDisplay::constructor
    #
    # arguments:
    #  parent: the parent widget
    #
    def __init__(self, parent=None):
        super(OutputDisplay, self).__init__(parent)

        # initialize the canvas and label objects
        #
        self.canvas = FigureCanvas(Figure(tight_layout=True))
        self.label = QtWidgets.QLabel()

        # set up the x and y coordinate arrays
        #
        self.x = np.empty((0, 0))
        self.y = np.empty((0, 0))

        # set up the covariance of the gaussian
        # when user draws
        #
        self.cov = [[0.05, 0.0], [0.0, 0.05]]
        self.num_points = 25

        # set up color variables: color bank, colors used, and
        # current colors
        #
        self.colors = [*cnames]
        self.colors_used = []
        self.color_c = None

        # set up current class, triggered class, all_classes
        #
        self.current_class = None
        self.t_current_class = None
        self.all_classes = []
        self.current_co = None

        # set up dictionary to store class information
        #
        self.once_c = True
        self.pair = 0
        self.class_info = {}

        # set up initial press and draw flag
        #
        self.pressed = False
        self.draw = 'point'

        # sets title and maximum size in pixels for Eval window
        #
        self.label.setText("Eval:")
        self.label.setStyleSheet("color: black;")
        self.label.setMaximumSize(35, 10)

        # defines the layout for the widget
        #
        vertical_layout = QtWidgets.QVBoxLayout(self)
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        vertical_layout.addWidget(self.label)
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.x_axis = [-DEFAULT, DEFAULT]
        self.y_axis = [-DEFAULT, DEFAULT]
        self.initUI()

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)

    # method: OutputDisplay::initUI
    #
    # This method creates the GUI.
    #
    def initUI(self):

        self.canvas.axes.set_xlim([self.x_axis[0], self.x_axis[-1]])
        self.canvas.axes.set_ylim([self.y_axis[0], self.y_axis[-1]])

        # set range and grid
        #
        self.canvas.axes.set_axisbelow(True)

        # set visual aspects of grid
        #
        self.canvas.axes.grid(which='major', color='grey', linestyle='-', linewidth=0.25, alpha=0.50)
        self.canvas.axes.tick_params(labelsize=8)

        self.x_axislim = self.canvas.axes.get_xlim()
        self.y_axislim = self.canvas.axes.get_ylim()
        self.x_axis = np.linspace(self.x_axislim[0], self.x_axislim[1], 9)
        self.y_axis = np.linspace(self.y_axislim[0], self.y_axislim[1], 9)
        self.canvas.axes.set_xticks(self.x_axis)
        self.canvas.axes.set_yticks(self.y_axis)

    # method: OutputDisplay::clear_plot
    #
    # This method clears any data that is shown in the output display.
    #
    def clear_plot(self):

        # clear the input canvas and reset with the intial UI
        #
        self.canvas.axes.clear()
        self.initUI()
        print(self.canvas.axes.collections)

        # iterate through class info and reset all information while keeping the color of the class
        #
        count = 0
        for key in self.class_info:
            # reset x and y coordinates
            #
            self.class_info[key][1] = np.empty((0, 0))
            self.class_info[key][2] = np.empty((0, 0))

            # reset class plot with same color
            #
            self.class_info[key][0] = self.canvas.axes.scatter(self.class_info[key][1], self.class_info[key][2], s=1)
            self.class_info[key][0].set_color(self.class_info[key][4])

            # set current class to the key
            #
            self.current_class = key
            self.class_info[self.current_class][0].set_gid(
                np.full((1, np.shape(self.class_info[self.current_class][1])[0]),
                        count))
            self.canvas.draw_idle()
            count = count + 1

        self.canvas.draw_idle()

    # method: OutputDisplay::clear_result_plot
    #
    # This method clears the results from the plot and keeps the data plotted
    #
    def clear_result_plot(self):

        # remove canvas and reinsert the initial UI
        #
        self.canvas.axes.clear()
        self.initUI()

        # iterate through class dictionary and replot the graphs without results
        #
        count = 0
        for key in self.class_info:
            self.class_info[key][0] = self.canvas.axes.scatter(self.class_info[key][1], self.class_info[key][2], s=1)
            self.class_info[key][0].set_color(self.class_info[key][4])
            self.current_class = key
            self.class_info[self.current_class][0].set_gid(
                np.full((1, np.shape(self.class_info[self.current_class][1])[0]),
                        count))
            count = count + 1
        self.canvas.draw_idle()

    # method: OutputDisplay::remove_class
    #
    # arguments:
    # name: the current class that is chosen to be removed
    #
    # This method removes a single class using the delete class button.
    #
    def remove_class(self, name):
        if self.class_info:
            # clear the visual plot
            #
            self.canvas.axes.clear()
            self.initUI()
            # set the current class and reset the colors used
            #
            tmp = self.current_class

            if self.class_info[self.current_class] is not None:
                color = self.class_info[self.current_class][4]
                if color in self.colors_used:
                    self.colors_used.remove(color)
                else:
                    self.colors.append(color)

                self.class_info.pop(self.current_class)

                # replot the remaining classes from class_info with their appropriate color
                #
                count = 0
                for key in self.class_info:
                    self.class_info[key][0] = self.canvas.axes.scatter(self.class_info[key][1], self.class_info[key][2],
                                                                       s=1)
                    self.class_info[key][0].set_color(self.class_info[key][4])
                    self.current_class = key
                    self.class_info[self.current_class][0].set_gid(
                        np.full((1, np.shape(self.class_info[self.current_class][1])[0]),
                                count))
                    self.canvas.draw_idle()
                    count = count + 1
                self.canvas.draw_idle()

            else:
                pass

    # method: OutputDisplay::on_press
    #
    # arguments:
    # event: the event loop for this app
    #
    # This method calculates where the mouse is on the display and records both the
    # the x and y coordinates of the data and stores them into a dictionary where
    # the class is linked with its data. There are two options either a click and drag
    # for points or use of gaussian plots. This makes up the evaluation data.
    #
    def on_press(self, event):

        # check if current class name is a string and if not make it a string
        #
        if not isinstance(self.current_class, str):
            self.current_class = "%s" % self.current_class

        # find the index of current class in total list
        #
        self.pair = self.t_current_class.index(self.current_class)

        # if the current class is not in the class dictionary or is initialized to Norn
        # initialize the class in the dictionary
        #
        if self.current_class not in self.class_info or (self.class_info[self.current_class] == None):
            if self.color_c == None:
                self.color_c = self.colors.pop()
            self.class_info[self.current_class] = [self.current_c, self.x, self.y, self.once_c, self.color_c]

        # set the draw to point and track the x and y data
        #
        if self.draw == "point":
            self.pressed = True

            if self.class_info[self.current_class][3]:
                self.class_info[self.current_class][0] = self.canvas.axes.scatter(None, None, s=1)
                self.class_info[self.current_class][3] = False
            self.class_info[self.current_class][1] = np.append(self.class_info[self.current_class][1], event.xdata)
            self.class_info[self.current_class][2] = np.append(self.class_info[self.current_class][2], event.ydata)

            self.class_info[self.current_class][0].set_color(self.class_info[self.current_class][4])
            self.canvas.draw_idle()

        # set the draw to gaussian and track the data produced by the target x,y and the
        # gaussian formula
        #
        else:
            self.pressed = True
            if self.class_info[self.current_class][3]:
                self.class_info[self.current_class][0] = self.canvas.axes.scatter(None, None, s=1)
                self.class_info[self.current_class][0].set_color(self.class_info[self.current_class][4])
                self.class_info[self.current_class][3] = False

            mu = [event.xdata, event.ydata]
            cov = self.cov
            gauss_plot = np.random.multivariate_normal(mu, cov, self.num_points)
            self.class_info[self.current_class][1] = np.append(self.class_info[self.current_class][1], gauss_plot[:, 0])
            self.class_info[self.current_class][2] = np.append(self.class_info[self.current_class][2], gauss_plot[:, 1])
            self.class_info[self.current_class][0].set_offsets(
                np.column_stack((self.class_info[self.current_class][1], self.class_info[self.current_class][2])))
            self.canvas.draw_idle()

    # method: OutputDisplay::on_move
    #
    # arguments:
    # event: the event loop for this app
    #
    # this method records the x and y data while the mouse is moving
    #
    def on_move(self, event):
        if self.pressed:

            # if set to point, append the x and y data of teh current mouse position
            #
            if self.draw == "point":
                self.class_info[self.current_class][1] = np.append(self.class_info[self.current_class][1], event.xdata)
                self.class_info[self.current_class][2] = np.append(self.class_info[self.current_class][2], event.ydata)
                self.class_info[self.current_class][0].set_offsets(
                    np.column_stack((self.class_info[self.current_class][1], self.class_info[self.current_class][2])))
                self.canvas.draw_idle()

            # if set to gaussian, record the gaussian with current mouse placement while moving
            #
            else:
                mu = [event.xdata, event.ydata]
                cov = self.cov
                gauss_plot = np.random.multivariate_normal(mu, cov, self.num_points)
                self.class_info[self.current_class][1] = np.append(self.class_info[self.current_class][1],
                                                                   gauss_plot[:, 0])
                self.class_info[self.current_class][2] = np.append(self.class_info[self.current_class][2],
                                                                   gauss_plot[:, 1])
                self.class_info[self.current_class][0].set_offsets(
                    np.column_stack((self.class_info[self.current_class][1], self.class_info[self.current_class][2])))
                self.canvas.draw_idle()

    # method: OutputDisplay::on_release
    #
    # arguments:
    # event: the event loop for this app
    #
    # reset the pressed flag and set a name for the class_info correlating with graphical data
    #
    def on_release(self, event):

        # set the pressed flag to false
        #
        self.pressed = False

        # set class in dictionary to pair with the graphical data
        #
        self.class_info[self.current_class][0].set_gid(np.full((1, np.shape(self.class_info[self.current_class][1])[0]),
                                                               self.pair))
        self.canvas.draw_idle()

    # method: OutputDisplay::find_class_c
    #
    # arguments:
    #  sender: The menu item that is currently triggered
    #
    # This method finds the current class object that is triggered and finds what
    # the appropriate color that is paired with with class
    #
    def find_class_c(self, sender=None):

        # check if the sender signal is triggered
        #
        if sender is not None:

            # find the class pair of the signal and set it to the current class
            #
            self.pair = self.all_classes.index(sender)
            self.current_co = sender
            self.current_class = self.t_current_class[self.pair]
        else:
            pass

    # method: OuputDisplay::set_point
    #
    # arguments: none
    #
    # set draw flag to point
    def set_point(self):
        self.draw = 'point'

    # method: OutputDisplay::set_gauss
    #
    # arguments: none
    #
    # set draw flag to gauss
    def set_gauss(self):
        self.draw = 'gauss'


# class: ProcessDescription
#
# arguments:
#  QWidget: base class for all UI interface objects
#
# This class contains methods that create the process description window which
# narrates the process of the algorithms as they run to evaluate the data
# in the Train and Eval windows.
#
class ProcessDescription(QtWidgets.QWidget):

    # method: ProcessDescription::constructor
    #
    # arguments:
    #  parent: the parent widget (widget that draws the child)
    #
    def __init__(self, parent=None):
        super(ProcessDescription, self).__init__(parent)

        # defines the size policy for the widget
        #
        self.output = QtWidgets.QTextEdit()

        self.label = QtWidgets.QLabel()
        self.pbar = QtWidgets.QProgressBar()

        self.initUI()

    def update_process(self, value):
        self.pbar.setValue = value

        self.value = value

    # method: ProcessDescription::initUI
    #
    # This method creates the GUI.
    #
    def initUI(self):
        # sets the textbox to read-only
        #
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.Monospace)
        self.output.setFont(font)

        self.output.setReadOnly(True)
        self.line_edit = QtGui.QTextLine()
        self.label.setText("Process Log:")

        # set size and color of title text
        #
        self.label.setStyleSheet("color: black;")
        self.label.setMaximumSize(124, 17)
        self.output.setStyleSheet("color: black;")

        # defines the layout for the widget
        #
        vertical_layout = QtWidgets.QVBoxLayout(self)
        vertical_layout.setContentsMargins(0, 0, 0, 0)

        vertical_layout.addWidget(self.label)
        vertical_layout.addWidget(self.pbar)
        vertical_layout.addWidget(self.output)

    # method: ProcessDescription::clear_progress
    #
    # arguments: None
    #
    def clear_progress(self):
        self.pbar.setValue(0)

#
# end of class

#
# end of file
