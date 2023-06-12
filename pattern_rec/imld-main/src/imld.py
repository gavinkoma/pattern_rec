#!/usr/bin/env python
#
# file: /data/isip/exp/demos/exp_0007/v0.0.6/imld.py
#
# file: imld.py
#
# revision history:
#
# 20200707 (LV): added missing functions, overhauled modules
# 20200305 (SJ): first version
#
# This file contains the main function of the ISIP Machine Learning Demo (IMLD)
# software.
#------------------------------------------------------------------------------

# import system modules
#
import sys
import logging
from PyQt5 import QtGui, QtWidgets, QtCore

# import local modules:
#  imld_gui_events: This is the main event handler loop
#
import gui.imld_gui_events as ige
import gui.imld_gui_window as igw

#------------------------------------------------------------------------------
#                                                                        
# functions are listed here                                                
#                                                                            
#------------------------------------------------------------------------------

# function: main
# 
# arguments: none
#
# return: none
#
def main():

    # create Qt boilerplate which performs event
    # handling, cursor handling and manages the application
    #
    app = QtWidgets.QApplication([])

    # create an event handler instance
    #
    handler = ige.EventHandler()

    # loop until an event is generated
    #
    app.exec_()

    # clean up and exit gracefully
    #
    sys.exit(0)

# begin gracefully
#
if __name__ == "__main__":

    #**> log all events that meet INFO level of severity
    #
    logging.basicConfig(level='INFO')

    # invoke the main program
    #
    main()

#
# end of file
