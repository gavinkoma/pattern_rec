#!/usr/bin/env python
#
# file: imld/data/imld_data_io.py
#                                                                              
# revision history:
#
# 20210923 (TC): clean up. Added write_data()
# 20200101 (..): initial version
#                                                                              
# This class contains a collection of functions that deal with data handling
# The structure of this class includes:
#     load_file() --> read_data()
#     save_file() --> write_data()
#
#------------------------------------------------------------------------------
#                                                                             
# imports are listed here                                                     
#                                                                             
#------------------------------------------------------------------------------

# import system modules
#
import numpy as np
from PyQt5 import QtWidgets

# import nedc modules
#
import lib.imld_constants_file as icf
# import gui.imld_gui_window as igw

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# strings used in menu prompts
#
LOAD_TRAIN_DATA = "Load Train Data"
LOAD_EVAL_DATA = "Load Eval Data"

SAVE_TRAIN_DATA = "Save Train Data as..."
SAVE_EVAL_DATA = "Save Eval Data as..."

# strings used in csv formatted files
#
FILE_HEADER_INFO = ["classes", "colors", "limits"]

# default filenames
#
DEFAULT_TRAIN_FNAME = "imld_train.csv"
DEFAULT_EVAL_FNAME = "imld_eval.csv"

# default file extensions
#
FILE_EXTS = ["csv"]
FILE_EXTS_STR = "Text files (*.csv)"

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

#  class: DataIO
#
#  This class contains methods to both save and load data for both
#  the train and eval windows of the application.
#
class DataIO:

    # method: DataIO::constructor
    #
    # arguments:
    #  usage: a short explanation of the command that is printed when 
    #         it is run without argument.
    #  help: a full explanation of the command that is printed when 
    #        it is run with -help argument.
    #
    # return: None
    #
    def __init__(self, ui ):
        
        # declare a data structure to hold user data
        #
        self.user_data = None

        # set the ui
        #
        self.ui = ui

        # exit gracefully
        #
        return None

    # method: DataIO::load_file
    #
    # arguments:
    #  mode: train or eval
    #
    # return:
    #  classes: list of classes from user input file
    #  colors: list of colors from user input file
    #  user_data: a dict of user data
    #
    def load_file(self, mode):

        # get mode, either train or eval
        #
        self.mode = mode

        # prompts user for file - displayed at the top of pop-up when user
        # clicks on File menu, depends on file being train or eval
        # check if user chooses train mode
        #
        if self.mode is icf.DEF_MODE[0]:

            # Reference:
            # static PySide2.QtWidgets.QFileDialog.getOpenFileName([parent=None[, 
            # caption=""[, dir=""[, filter=""[, selectedFilter=""[, 
            # options=QFileDialog.Options()]]]]]])
            # See more: https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QFileDialog.html#PySide2.QtWidgets.PySide2.QtWidgets.QFileDialog.getOpenFileName
            #
            file,_ = QtWidgets.QFileDialog.getOpenFileName(
                                                    self.ui,
                                                    LOAD_TRAIN_DATA,
                                                    icf.DELIM_NULL,)

        # check if user chooses eval mode
        #
        elif self.mode is icf.DEF_MODE[1]:

            # Reference:
            # Same as imld_data_io.load_file()
            #
            file,_ = QtWidgets.QFileDialog.getOpenFileName(
                                                    self.ui,
                                                    LOAD_EVAL_DATA,
                                                    icf.DELIM_NULL,)

        # covert file into string for parsing
        #
        file = str(file)

        # make sure a file was selected
        #
        if file is None or len(file) == 0:
            return None, None

        # read data
        #
        classes, colors, limits, self.user_data = self.read_data(file)

        # check if data was read
        #
        if self.user_data is None:
            print("Warning: Data not read or updated")
            return False

        # exit gracefully
        #
        return classes, colors, limits, self.user_data

    #
    # end of method

    # method: DataIO::save_file
    #
    # arguments:
    #  data: a dict of data wanted to save
    #  mode: train or eval
    #
    # return: None
    #
    def save_file(self, data, mode, limits):

        # prompts user for name to save file as, depending on data
        #
        if mode is icf.DEF_MODE[0]:

            # Reference:
            # static PySide2.QtWidgets.QFileDialog.getSaveFileName([parent=None[, 
            # caption=""[, dir=""[, filter=""[, selectedFilter=""[, 
            # options=QFileDialog.Options()]]]]]])
            # See more: https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QFileDialog.html#PySide2.QtWidgets.PySide2.QtWidgets.QFileDialog.getSaveFileName
            #
            save_name = QtWidgets.QFileDialog.getSaveFileName(self.ui,
                                                              SAVE_TRAIN_DATA,
                                                              DEFAULT_TRAIN_FNAME,
                                                              FILE_EXTS_STR)
        elif mode is icf.DEF_MODE[1]:

            # Reference:
            # Same as imld_data_io.save_file()
            #
            save_name = QtWidgets.QFileDialog.getSaveFileName(self.ui,
                                                          SAVE_EVAL_DATA,
                                                          DEFAULT_EVAL_FNAME,
                                                          FILE_EXTS_STR)
        else:
            pass

        # make sure a file name was given
        #
        if save_name[0] is icf.DELIM_NULL or len(save_name) == 0:
            return

        # append file extension if user save name doesn't include exts
        #
        if save_name[0][-4:] != (icf.DELIM_DOT+FILE_EXTS[0]):
            save_name = save_name[0] + (icf.DELIM_DOT+FILE_EXTS[0])
        else:
            save_name = save_name[0]

        self.write_data(data, save_name, limits)

    # method: DataIO::read_data
    #
    # arguments:
    #  fname: input filename
    #
    # return:
    #  classes: list of classes from user input file
    #  colors: list of colors from user input file
    #  user_data: a dict of user data
    #
    @staticmethod
    def read_data(fname):

        classes = []
        colors = []
        limits = []
        data = []

        # open file
        #
        with open(fname, icf.MODE_READ_TEXT) as fp:
            
            # loop over lines in file
            #
            for num_line, line in enumerate(fp):
                
                # clean up the line
                #
                line = line.replace(icf.DELIM_NEWLINE, icf.DELIM_NULL) \
                           .replace(icf.DELIM_CARRIAGE, icf.DELIM_NULL)
                check = line.replace(icf.DELIM_SPACE, icf.DELIM_NULL)

                # get classes in csv file
                #
                if check.startswith(icf.DELIM_COMMENT + FILE_HEADER_INFO[0] ):
                    
                    # get classes after colon
                    #
                    check = check.split(icf.DELIM_COLON)[1]\
                            .replace(icf.DELIM_OPEN, icf.DELIM_NULL)\
                            .replace(icf.DELIM_CLOSE, icf.DELIM_NULL)

                    # split to list
                    #
                    classes = check.split(icf.DELIM_COMMA)
                    
                    continue

                # get colors in csv file
                #
                if check.startswith(icf.DELIM_COMMENT + FILE_HEADER_INFO[1]):

                    # get colors after colon
                    #
                    check = check.split(icf.DELIM_COLON)[1].replace(icf.DELIM_OPEN, icf.DELIM_NULL) \
                            .replace(icf.DELIM_CLOSE, icf.DELIM_NULL)

                    # split to list
                    #
                    colors = check.split(icf.DELIM_COMMA)

                    continue

                # get limits in csv file
                #
                if check.startswith(icf.DELIM_COMMENT + FILE_HEADER_INFO[2]):

                    # get limits after colon
                    #
                    check = check.split(icf.DELIM_COLON)[1].replace(icf.DELIM_OPEN, icf.DELIM_NULL) \
                        .replace(icf.DELIM_CLOSE, icf.DELIM_NULL)

                    # split to list
                    #
                    limits = check.split(icf.DELIM_COMMA)

                    continue

                # get data
                #
                if not check.startswith(icf.DELIM_COMMENT):
                    try:
                        class_name, x, y = check.split(icf.DELIM_COMMA)[0:3]
                        data.append([class_name, float(x), float(y)])
                    except:
                        print("Error loading at line %d" % num_line)

                    continue
            #
            # end of for

            # convert list of data to a dict
            #
            user_data = {}
            for item in data:
                if item[0] not in user_data:
                    user_data[item[0]] = []

                user_data[item[0]].append([np.array(item[1]), np.array(item[2])])
            
        #
        # end of with open file

        # exit gracefully
        #
        return classes, colors, limits, user_data
    #
    # end of method

    # method: DataIO::write_data
    #
    # arguments:
    #  data: data wanted to be save. The structure is same as class_info.
    #      data = {classname: [[...], [X], [Y], [...], [color]], etc}
    #  fname: filename wanted to be save as
    #
    # return: None
    #
    @staticmethod
    def write_data(data, fname, limits):

        # creates the file and writes the data to it
        #
        with open(fname, icf.MODE_WRITE_TEXT) as fp:

            # import all class_info from InputDisplay
            #
            class_info = data

            # get list of colors
            #
            colors = []
            for class_name in class_info:
                colors.append(class_info[class_name][4])

            colors =  icf.DELIM_OPEN \
                        + icf.DELIM_COMMA.join(str(e) for e in colors) \
                        + icf.DELIM_CLOSE

            classes = icf.DELIM_OPEN \
                        + icf.DELIM_COMMA.join(str(e) for e in \
                                               list(class_info.keys())) \
                        + icf.DELIM_CLOSE
            limits = icf.DELIM_OPEN + icf.DELIM_COMMA.join(str(e) for e in limits) + icf.DELIM_CLOSE

            # write comment with classes and colors lists
            #
            fp.write("# filename: %s\n" % fname)
            fp.write("# classes: %s\n" % classes)
            fp.write("# colors: %s\n" % colors)
            fp.write("# limits: %s\n" % limits)
            fp.write("# \n")
            
            for class_name in class_info:

                # retrieve class coordinate points
                #
                data_x = class_info[class_name][1]
                data_y = class_info[class_name][2]
                data_t = np.column_stack((data_x,data_y))

                # write each point of each class in a csv file
                #
                for item in range(len(data_t)):

                    # write "classname,""
                    #
                    fp.write(class_name + icf.DELIM_COMMA)

                    # write "x,y"
                    #
                    for num in range(len(data_t[item])):
                        if data_t[item][num] == data_t[item][-1]:
                            fp.write("%8lf" % (data_t[item][num]))
                        else:
                            fp.write("%8lf," % (data_t[item][num]))

                    # write new line
                    #
                    fp.write(icf.DELIM_NEWLINE)

        # close file
        #
        fp.close()

        # exit gracefully
        #
        return True

    #
    # end of method

#
# end of class

#
# end of file
