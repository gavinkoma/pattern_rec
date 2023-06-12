File: imld/v1.8.0/AAREADME.txt
Tool: ISIP Machine Learning Demo
Version: 1.8.0
#IMLD 1.8.0
-------------------------------------------------------------------------------
Change Log:

(20220407) updated README for v1.8.0

-------------------------------------------------------------------------------

This directory contains all the Python code needed to run our learning 
demo tool. This tool is used to aid in learning machine learning topics.  

A. WHAT'S NEW

Version 1.8.0 includes these enhancements:

 - Implemented Progress Bar
 - Implemented Normalizer option
 - Fixed reported bugs


B. INSTALLATION REQUIREMENTS

Python code unfortunately often depends on a large number of add-ons, making
it very challenging to port code into new environments. This tool has been
tested extensively on Windows and Mac machines running Python v3.7.x.

Software tools required include:

-  Python 3.7.x or higher (we recommend installing Anaconda)
-  PyQt5: https://www.riverbankcomputing.com/software/pyqt/download5
-  Numpy/SciPy: http://www.numpy.org/
-  PyQtGraph: http://www.pyqtgraph.org/ (v0.12.1 or higher)


These dependencies can be installed using pip:

-  pip install pyqt5
-  pip install pyqtgraph
-  pip install scipy
-  pip install matplotlib


Or by using the requirements.txt and running:

pip install -r requirements.txt

For Mac users, since Mac OS X 10.8 comes with Python 2.7, you may 
need to utilize pip3 when attempting to install dependencies:

-  pip3 install pyqt5
-  pip3 install pyqtgraph
-  pip3 install scipy
-  pip3 install matplotlib



C. USER'S GUIDE

The easiest way to run this is to run imld.py which is found in the imld/src folder.


Best regards,

Joe Picone
