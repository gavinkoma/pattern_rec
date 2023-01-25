#!/usr/bin/env python
#
# file: imld/src/lib/imld_constants_datagen.py
#
# revision history:
#  20210923 (TC): initial version
#
# usage:
#  import imld_constants_datagen as icd
#
# This class contains a collection of variables used for data generation
#------------------------------------------------------------------------------
#
# imports are listed here
#
#------------------------------------------------------------------------------

# import system modules
#
import os
import re
import sys
import errno

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# strings used in classes for all datasets
#
DEFAULT_CLASSES = ["Class0", "Class1", "Class2", "Class3"]
DEFAULT_NUM_CLASSES = len(DEFAULT_CLASSES)
DEFAULT_NPTS_PER_CLASS = 10000

# default Gaussian parameters
#
DEFAULT_TWOGAUSSIAN_MEAN = [[-0.5000, 0.5000], [0.5000, -0.5000]]
DEFAULT_TWOGAUSSIAN_COV = [[[0.0250,0.0000],[0.0000, 0.0250]],
                           [[0.0250,0.0000],[0.0000, 0.0250]]]

DEFAULT_OVERLAPGAUSSIAN_MEAN = [[-0.1400, 0.1400], [0.1400, 0.1400]]
DEFAULT_OVERLAPGAUSSIAN_COV = [[[0.0250,0.0000],[0.0000, 0.0250]],
                               [[0.0250,0.0000],[0.0000, 0.0250]]]

DEFAULT_FOURGAUSSIAN_MEAN = [[-0.5000, 0.5000], [0.5000, -0.5000],
                             [-0.5000, -0.5000], [0.5000, 0.5000]]
DEFAULT_FOURGAUSSIAN_COV = [[[0.0250,0.0000],[0.0000, 0.0250]],
                            [[0.0250,0.0000],[0.0000, 0.0250]],
                            [[0.0250,0.0000],[0.0000, 0.0250]],
                            [[0.0250,0.0000],[0.0000, 0.0250]]]

# default Ellipse parameters
#
DEFAULT_TWOELLIPSE_MEAN = [[-0.5000, 0.5000], [0.5000, -0.5000]]
DEFAULT_TWOELLIPSE_COV = [[[0.0333,0.0000],[0.0000, 0.0043]],
                          [[0.0333,0.0000],[0.0000, 0.0043]]]

DEFAULT_FOURELLIPSE_MEAN = [[-0.5000, 0.5000], [0.5000, -0.5000],
                            [-0.5000, -0.5000], [0.5000, 0.5000]]
DEFAULT_FOURELLIPSE_COV = [[[0.0333,0.0000],[0.0000, 0.0043]],
                           [[0.0333,0.0000],[0.0000, 0.0043]],
                           [[0.0333,0.0000],[0.0000, 0.0043]],
                           [[0.0333,0.0000],[0.0000, 0.0043]]]

DEFAULT_ROTATEDELLIPSE_MEAN = [[-0.5000, 0.5000], [0.5000, -0.5000]]
DEFAULT_ROTATEDELLIPSE_COV = [[[0.0333,0.0000],[0.0000, 0.0043]],
                              [[0.0043,0.0000],[0.0000, 0.0333]]]

# default Toroidal parameters
#
DEFAULT_TOROID_NUM_CLASSES = 2
DEFAULT_TOROID_NPTS_RING = 2000
DEFAULT_TOROID_OUTER_RADIUS = 0.8500
DEFAULT_TOROID_INNER_RADIUS = 0.6500
DEFAULT_TOROID_INNER_MEAN = [0.0000,0.0000]
DEFAULT_TOROID_INNER_COV = [[0.0083,0.0000],[0.0000, 0.0083]]

#default Ying Yang Parameters
#
DEFAULT_YING_NPTS = 2000
DEFAULT_YANG_NPTS = 2000
DEFAULT_YINGYANG_OVERLAP = 0.1000

#
# end of file

