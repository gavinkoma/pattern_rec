#!/usr/bin/env python
#
# file: imld/src/lib/imld_constants_file.py
#                                                                              
# revision history:
#  20210923 (TC): initial version
#                                                                              
# usage:
#  import imld_constants_file as icf
#                                                                              
# This class contains a collection of functions that deal with file handling
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

# set the default character encoding system
#
DEF_CHAR_ENCODING = "utf-8"

# file processing charater constants
#
DELIM_BLANK = '\x00'
DELIM_BOPEN = '{'
DELIM_BCLOSE = '}'
DELIM_CARRIAGE = '\r'
DELIM_CLOSE = ']'
DELIM_COLON = ':'
DELIM_COMMA = ','
DELIM_COMMENT = '#'
DELIM_DASH = '-'
DELIM_DOT = '.'
DELIM_EQUAL = '='
DELIM_NEWLINE = '\n'
DELIM_NULL = ''
DELIM_OPEN = '['
DELIM_QUOTE = '"'
DELIM_SEMI = ';'
DELIM_SLASH = '/'
DELIM_SPACE = ' '
DELIM_SQUOTE = '\''
DELIM_TAB = '\t'
DELIM_USCORE = '_'

# file processing string constants
#
STRING_EMPTY = ""
STRING_DASHDASH = "--"

# file processing lists:
#  used to accelerate some functions
#
LIST_SPECIALS = [DELIM_SPACE, DELIM_BLANK]

# i/o constants
#
MODE_READ_TEXT = "r"
MODE_READ_BINARY = "rb"
MODE_WRITE_TEXT = "w"
MODE_WRITE_BINARY = "wb"

# imld mode of dataset
#
DEF_MODE = ['train', 'eval']

# model random seed
#
SEED = 45
#                                                                              
# end of file 

