#!/usr/bin/env python
#
# file: $NEDC_NFC/class/python/nedc_ann_tools/nedc_ann_dpath_tools.py
#                                                                              
# revision history: 
# 
# 20220406 (PM): Write function
# 20220303 (PM): Modified AnnDpath XML section to support schema 
# 20220128 (ML): added  functions to support csv
# 20220126 (JP): completed another code review
# 20220122 (PM): Updated the API for the return structure of read
# 20220117 (JP): completed the first pass of code review
# 20220112 (PM): Add the CSV class
# 20220106 (PM): modified the XML class
# 20211229 (PM): added DpathChecker Class
# 20210201 (TC): initial version
#                                                                              
# This class contains a collection of methods that provide 
# the infrastructure for processing annotation-related data.
#------------------------------------------------------------------------------

# import reqired system modules
#
import os
import pandas
import pprint
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path as path
from lxml import etree
from xml.dom import minidom

# import required NEDC modules
#
import nedc_debug_tools as ndt
import nedc_file_tools as nft

#------------------------------------------------------------------------------
#                                                                              
# global variables are listed here                                             
#                                                                              
#------------------------------------------------------------------------------

# set the filename using basename                                              
#                                                                              
__FILE__ = os.path.basename(__file__)

# define the string to check the file header
#
DEF_CSV_VERSION = "# version = csv_v1.0.0"
DEF_CSV_EXT = nft.DEF_EXT_CSV

# define default xml schema
#
DEF_XML_SCHEMA = "$NEDC_NFC/lib/nedc_dpath_xml_schema_v00.xsd"

# define xml parsing-related variables
#
REGION_PATH = "Annotation/Regions/Region"
REGION_ATTR_PATH = "Attributes/Attribute"
ATTR_PATH = "Annotation/Attributes/Attribute"
CONFIDENCE = 1.0000

# define constants for printing the xml information and accessing the API 
# dictionary
#
DEF_LABEL_COORDS = nft.DEF_XML_COORDS
DEF_LABEL_CONFIDENCE = nft.DEF_XML_CONFIDENCE
DEF_LABEL_TEXT = nft.DEF_XML_TEXT
DEF_LABEL_REGION_ID = nft.DEF_XML_REGION_ID
DEF_LABEL_TISSUE_VALUE = nft.DEF_XML_TISSUE_TYPE
DEF_LABEL_TISSUE = 'tissue'
DEF_LABEL_NULL = "null"

# define constants for accessing the XML attribute with ElementTree
#
DEF_ATTR_LABEL = nft.DEF_XML_LABEL
DEF_ATTR_RINDEX = nft.DEF_XML_REGION_ID
DEF_ATTR_WIDTH = nft.DEF_XML_WIDTH
DEF_ATTR_HEIGHT = nft.DEF_XML_HEIGHT
DEF_ATTR_CONFIDENCE = nft.DEF_XML_CONFIDENCE.capitalize()
DEF_ATTR_TEXT = nft.DEF_XML_TEXT.capitalize()
DEF_ATTR_AREA = "Area"
DEF_ATTR_AREAM = "AreaMicrons"
DEF_ATTR_ATTR = "Attribute"
DEF_ATTR_ID = "Id"
DEF_ATTR_LENGTH = "Length"
DEF_ATTR_LENGTHM = "LengthMicrons"
DEF_ATTR_NAME = "Name"
DEF_ATTR_BNAME = "bname"
DEF_ATTR_MICRONS = "MicronsPerPixel"
DEF_ATTR_ROW = "row"
DEF_ATTR_COLUMN = "column"
DEF_ATTR_DEPTH = "depth"
DEF_ATTR_INDEX = "index"
DEF_ATTR_REGION = "Region"
DEF_ATTR_VALUE = "Value"
DEF_ATTR_VERTEX = "Vertex"
DEF_ATTR_VERTICES = "Vertices"
DEF_ATTR_TISSUE_VALUE = "Tissue Value"
DEF_ATTR_X = "X"
DEF_ATTR_Y = "Y"
DEF_ATTR_Z = "Z"

# define output formats
#
DEF_FMT_REGION = " Region %s:\n"
DEF_FMT_DIMEN = " width = %s pixels, height = %s pixels\n"
DEF_FMT_HEAD = " %s = %s\n"
DEF_FMT_ITEM = "  %s = %s\n"
DEF_FMT_VERTS = "  %s: min_x = %s, max_x = %s, min_y = %s, max_y = %s\n"
DEF_FMT_HEADER = "index,region_id,tissue,label,coord_index," + \
               "row,column,depth,confidence"

# define the regular expression
#
DEF_REGEX_COMMENT = re.compile(f'(# [a-z].+?(?=\n))', re.IGNORECASE)
DEF_REGEX_MICRON = re.compile(f'(MicronsPerPixel) = (\d+.\d+)')
DEF_REGEX_WIDTH = re.compile(f'(width) = (\d+)')
DEF_REGEX_HEIGHT = re.compile(f'(height) = (\d+)')

# declare a global debug object so we can use it in functions and classes
#                                                                              
dbgl = ndt.Dbgl()

#------------------------------------------------------------------------------
#
# functions listed here
#
#------------------------------------------------------------------------------

# method: read
#
# arguments:
#  fname: the file to be processed
#
# return: a dictionary representation of the parsed file if a valid type
#
# This method reads an annotation file from disk. 
# 
# The form of the dictionary is:
#
#  header: {
#     'MicronsPerPixel' : microns_value (str),
#     'height' : height_value (str),
#     'width' : width_value (str)
#  }
#
#  data: {
#     0: {
#        'confidence' : confidence_value (str),
#        'coordinates': [[x1,y1,z1], [x2,y2,z2], ...] (List[list[int]]),
#        'region_id' : id_value (str),
#        'text' : text_value (str),
#        'tissue_type' : tissue_type_value (str)
#     },
#     n: { ... }
#  }
#
#
def read(fname):

    # declare local variables
    #
    ann = AnnDpath()
    xml = Xml()
    csv = Csv()
    header = {}
    data = {}

    # display debug information
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: reading file %s" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
    
    # branch on the file type: xml
    #
    if ann.is_xml(fname):
        header, data = xml.read(fname)
        if header and data is False:
            print("Error: %s (line: %s) %s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return None
    
    # branch on the file type: csv
    #
    elif ann.is_csv(fname):
        header, data = csv.read(fname)
        if header and data is False:
            print("Error: %s (line: %s) %s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return None

    # unknown file type: error
    #
    else:
        print("Error: %s (line: %s) %s: unknown file type (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
        return None

    # exit gracefully
    #
    return header, data

# method: write 
#
#
def write(fname, oname, type = None):
    
    # declare local variables
    #
    csv = Csv()

    if type == "csv":
        csv.write_header(fname, fp = oname)
        csv.write_data(fname, fp = oname)
    elif type == "xml":
        print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "Does not support CSV to XML Convertion", fname))
        sys.exit(os.EX_SOFTWARE)
    else:
        print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "Invalid Dpath file type", fname))
        sys.exit(os.EX_SOFTWARE)
    
    return True

# method: print_events_from_file
#
# arguments:
#  fname: the file to be processed
#  fp: the output file pointer
#
# return: a boolean value indicating status
#
# This method displays annotation events in a human readable format.
#
def print_events_from_file(fname, fp = sys.stdout):
    
    # declare local variables
    #
    ann = AnnDpath()
    xml = Xml()
    csv = Csv()

    # display debugging information
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: printing events for %s" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
    
    # branch on the file type: xml
    #
    if ann.is_xml(fname) is True:
        if xml.print_events_from_file(fname, fp) is False:
            print("Error: %s (line: %s) %s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False
    
    elif ann.is_csv(fname) is True:
        if csv.print_events_from_file(fname, fp) is False:
            print("Error: %s (line: %s) %s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False

    # unknown file type: error
    #
    else:
        print("Error: %s (line: %s) %s: unknown file type (%s)" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
        return False

    # exit gracefully
    #
    return True

#------------------------------------------------------------------------------
#
# classes are listed here:
#
#------------------------------------------------------------------------------

# class: AnnDpath
#
# This class is the main class of this file. It contains methods to 
# manipulate the set of supported annotation file formats (xml/csv).
#
class AnnDpath:

    # method: AnnDpath::constructor
    #
    # arguments: none
    #
    # return: none
    #
    # This method constructs an AnnDpath object.
    #
    def __init__(self, *, xml_schema = DEF_XML_SCHEMA):
        
        # set the class name
        #
        AnnDpath.__CLASS_NAME__ = self.__class__.__name__
        
        # set the schema
        #
        self.xml_schema = nft.get_fullpath(xml_schema)

        # display debug information
        #
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s::%s: contructing an annotation object" %
                  (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                   ndt.__NAME__))

        #
        # end of method

    # method: AnnDpath::is_ann
    #
    # arguments: 
    #  fname: the file name
    # 
    # return: a boolean value indicating status
    #
    # A general method that checks the validity of the input file.
    #
    def is_ann(self, fname):

        # declare local variables
        #
        status = False

        # display debugging information
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: checking is_ann (%s)" %
                  (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                   ndt.__NAME__, fname))

        # skip any list file 
        #
        if fname[-5:] == ".list":
            return

        # branch on the file type: xml
        #
        if self.is_xml(fname):
            status = True
        elif self.is_csv(fname):
            status = True

        # exit gracefully
        #
        return status
        
    # method:: AnnDpath::is_xml
    #
    # arguments:
    #   fname: the file name
    # 
    # return: a boolean value indicating status
    #
    # This method returns True if the parsed xml matches the schema.
    #
    def is_xml(self, fname):

        # declare validator
        #
        xml_validator = etree.XMLSchema(file=self.xml_schema)

        # display debugging information
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: checking for xml (%s)" %
                  (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                   ndt.__NAME__, fname))

        # parse the file
        #
        try:
            fp = etree.parse(fname)
        except OSError:
            print("Error: %s (line: %s) %s::%s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                   ndt.__NAME__, fname))
            return False
        except etree.XMLSyntaxError:
            if dbgl > ndt.BRIEF:
                print("Error: %s (line: %s) %s::%s: (%s) might not be an XML file" %
                  (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                   ndt.__NAME__, fname))
            return False

        # validate the xml file to the schema
        #
        if xml_validator.validate(fp):
            return True
        else:
            if dbgl > ndt.BRIEF:
                print("Error: %s (line: %s) %s::%s: processing error (%s)" %
                      (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                       ndt.__NAME__, fname))
            return False
        #
        # end of method

    # method:: AnnDpath::is_csv
    #
    # arguments:
    #  fname: the file name
    # 
    # return: a boolean value indicating status
    #
    # This method returns True if the metadata is a valid csv header.
    #
    def is_csv(self, fname):

        # display debugging information
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: checking for csv (%s)" %
                  (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                   ndt.__NAME__, fname))

        # open the file
        #
        fp = open(fname, nft.MODE_READ_TEXT)
        if fp is None:
            print("Error: %s (line: %s) %s::%s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                   ndt.__NAME__, fname))
            return False
        
        # read the first line in the file
        #
        header = fp.readline()
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: header (%s)" %
                  (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                   ndt.__NAME__, header))
        fp.close()

        # exit gracefully:
        #  if the beginning of the file is the magic sequence
        #  then it is an imagescope xml file
        if DEF_CSV_VERSION in header.strip():
            return True
        else:
            if dbgl > ndt.BRIEF:
                print("Error: %s (line: %s) %s::%s: processing error (%s)" %
                      (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                       ndt.__NAME__, fname))
            return False

        #
        # end of method

    # method:: AnnDpath::set_xml_schema
    #
    # arguments:
    #   new_xml_schema: the new xml schema to set it
    #
    # return: None
    #
    # This function sets the global schema variable
    #
    def set_xml_schema(self, new_xml_schema):

        self.xml_schema = nft.get_fullpath(new_xml_schema)
        if dbgl > ndt.BRIEF:
                print("%s (line: %s) %s::%s: New XML Schema = %s" %
                      (__FILE__, ndt.__LINE__, AnnDpath.__CLASS_NAME__,
                       ndt.__NAME__, self.xml_schema))
    #
    # end of method
#
# end of class
                                    
# class: Xml
#
# This class abstracts xml processing.
#
class Xml:

    # root of the file
    #
    root_d = None
    
    # method Xml::constructor
    # 
    # argument: None
    #  
    #
    # This is a constructor for the xml class.
    #
    def __init__(self):

        # set the class name
        #
        Xml.__CLASS_NAME__ = self.__class__.__name__

        #
        # end of method

    # method Xml::read
    #
    # arguments:
    #  fname: filename
    #  confidence: confidence value
    #
    # return: two dictionaries: header and data
    #
    # This method takes the filename and returns two dictionaries
    # containing the xml header and data. The form of the dictionary is:
    #
    #  header: {
    #     'MicronsPerPixel' : microns_value (str),
    #     'height' : height_value (str),
    #     'width' : width_value (str)
    #  }
    #
    #  data: {
    #     0: {
    #        'confidence' : confidence_value (str),
    #        'coordinates': [[x1,y1,z1], [x2,y2,z2], ...] (List[list[int]]),
    #        'region_id' : id_value (str),
    #        'text' : text_value (str),
    #        'tissue_type' : tissue_type_value (str)
    #     },
    #     n: { ... }
    #  }
    #
    def read(self, fname,
             confidence = CONFIDENCE):

        # display debugging information
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: reading xml file (%s)" %
                  (__FILE__, ndt.__LINE__, Csv.__CLASS_NAME__,
                   ndt.__NAME__, fname))

        # declare local variable and dictionaries
        #
        root_d = ET.parse(fname).getroot()
        header = {}
        data = {}

        # adds the "header" key to the dict using dictionary comphrehension
        #
        header = {attri.get(DEF_ATTR_NAME).lower():
                  attri.get(DEF_ATTR_VALUE).lower()
                  for attri in root_d.findall(ATTR_PATH)}
        header[DEF_ATTR_MICRONS] = root_d.attrib[DEF_ATTR_MICRONS]

        # loop through each region and add it to the dictionary by index 
        #
        for indx, reg in enumerate(root_d.findall(REGION_PATH)):

            # initialize a dictionary and a list
            #
            data[indx] = {}
            coordinate = []
            
            # get the coordinate
            #
            for vertex in reg.iter(DEF_ATTR_VERTEX):
                coordinate.append([int(float(vertex.attrib[DEF_ATTR_X])), 
                                    int(float(vertex.attrib[DEF_ATTR_Y])), 
                                    int(float(vertex.attrib[DEF_ATTR_Z]))])

            # assign them to the data dictionary 
            #
            data[indx][DEF_LABEL_REGION_ID] = reg.get(DEF_ATTR_ID)
            data[indx][DEF_LABEL_TEXT] = reg.get(DEF_ATTR_TEXT)
            data[indx][DEF_LABEL_COORDS] = coordinate
            data[indx][DEF_LABEL_TISSUE_VALUE] = \
                reg.find(REGION_ATTR_PATH).get(DEF_ATTR_VALUE)
            data[indx][DEF_LABEL_CONFIDENCE] = str(confidence)
           
        # return the dictionaries
        #
        return (header, data)
    
    # method Xml::print_events_from_file
    #
    # arguments:
    #  fname: file name
    #  fp: file pointer
    # 
    # return: a boolean value indicating status
    #
    # This method pretty prints the xml annotation information.
    #
    def print_events_from_file(self, fname, fp = sys.stdout):

        # parse the xml file
        #
        root_d = ET.parse(fname).getroot()

        if dbgl > ndt.BRIEF:
            fp.write("%s (line: %s) %s: parsing (%s)\n" %
                     (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

        # get the dimension
        #
        width, height = [val.attrib[DEF_ATTR_VALUE]
                         for val in root_d.findall(ATTR_PATH)]

        # print the header
        #
        fp.write(DEF_FMT_HEAD %
                 (DEF_ATTR_MICRONS, (root_d.attrib[DEF_ATTR_MICRONS])))
        
        # Note: width = dimension[0], height = dimension[1]
        #
        fp.write(DEF_FMT_DIMEN %
                    (width, height))

        # print the region's information
        #
        for child in root_d.iter(DEF_ATTR_REGION):

            x_vertices = [x.attrib[DEF_ATTR_X]
                          for x in child.iter(DEF_ATTR_VERTEX)]
            y_vertices = [y.attrib[DEF_ATTR_Y]
                          for y in child.iter(DEF_ATTR_VERTEX)]
            values = ["".join(value.attrib[DEF_ATTR_VALUE])
                      for value in child.iter(DEF_ATTR_ATTR)]

            fp.write(DEF_FMT_REGION % 
                     (child.attrib[DEF_ATTR_ID]))
            fp.write(DEF_FMT_ITEM %
                     (DEF_ATTR_TISSUE_VALUE, values[0]))
            fp.write(DEF_FMT_ITEM %
                     (DEF_ATTR_LENGTH, child.attrib[DEF_ATTR_LENGTH]))
            fp.write(DEF_FMT_ITEM %
                     (DEF_ATTR_AREA, child.attrib[DEF_ATTR_AREA]))
            fp.write(DEF_FMT_ITEM %
                     (DEF_ATTR_LENGTHM, child.attrib[DEF_ATTR_LENGTHM]))
            fp.write(DEF_FMT_ITEM %
                     (DEF_ATTR_AREAM, child.attrib[DEF_ATTR_AREAM]))
            fp.write(DEF_FMT_ITEM %
                     (DEF_ATTR_TEXT, child.attrib[DEF_ATTR_TEXT]))
            fp.write(DEF_FMT_VERTS % 
                     (DEF_ATTR_VERTICES, min(x_vertices), max(x_vertices),
                      min(y_vertices), max(y_vertices)))

        # exit gracefully
        #
        return True

#
# end of class

# class: Csv
#
# This class abstracts csv processing.
# 
class Csv:

    # method Csv::constructor
    # 
    # argument: None
    #
    # This is a constructor for the Csv class.
    #
    def __init__(self):

        # set the class name
        #
        Csv.__CLASS_NAME__ = self.__class__.__name__

        #
        # end of method

    # method Csv::read
    # 
    # arguments:
    #  fname: filename
    #
    # return: two dictionaries of the parsed csv data (header and data)
    #

    # This method takes the filename and returns two dictionaries
    # containing the csv header and data. The form of the dictionary
    # is the same as that for Xml::read.
    #

    # Note: The return dictionaries mirror the XML::read method so
    # both have the same information and structure
    #
    def read(self, fname):

        # declare local variable
        #
        header = {}
        data = {}
        f_header = []
        tissue_type = []
        text = []
        coordinates = []

        # open a file
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: opening (%s)\n" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

        file = open(fname, nft.MODE_READ_TEXT)
        if file is None:
            print("Error: %s (line: %s) %s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False

        # get the header of the file using regular expression
        #
        for line in file:
            if re.match(DEF_REGEX_COMMENT, line):
                f_header.append(
                    line.strip(nft.DELIM_COMMENT).strip(nft.DELIM_NEWLINE))
        
        # close the file and reset the pointer
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: closing (%s)\n" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
        file.close()

        # appending to the header dictionary
        #
        for f_line in f_header:
            height = re.findall(DEF_REGEX_HEIGHT, f_line)
            width = re.findall(DEF_REGEX_WIDTH, f_line)
            micron = re.findall(DEF_REGEX_MICRON, f_line)
            if micron:
                header[micron[0][0]] = micron[0][1]
            if height and width:
                header[height[0][0]] = height[0][1]
                header[width[0][0]] = width[0][1]

        # read the csv file with Pandas
        #
        parsed_csv = pandas.read_csv(fname, header = 0,
                                     comment = nft.DELIM_COMMENT,
                                     skip_blank_lines = True)

        # display debugging information
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: parsed csv file:\n %s" %
                  (__FILE__, ndt.__LINE__, Csv.__CLASS_NAME__,
                   ndt.__NAME__, parsed_csv))
        
        # convert the parsed csv to a dictionary
        #
        region = pandas.DataFrame.to_dict(parsed_csv, orient="list")

        # count the number of lines for each region
        #
        region_counts = [region[DEF_ATTR_RINDEX].count(i) 
                         for i in set(region[DEF_ATTR_RINDEX])]

        # add the coordinates for each region in a list
        # append the tag name in a list
        #
        counter = 0
        for num_reg in region_counts:
            
            # temporary list for each region
            #
            temp_coord = []
            text.append(region[DEF_ATTR_LABEL][counter])
            tissue_type.append(region[DEF_LABEL_TISSUE][counter])
            
            # add each vertices to the appropriate temporary list
            #
            for i in range(counter, num_reg + counter):
                temp_coord.append([int(float(region[DEF_ATTR_ROW][i])), 
                                   int(float(region[DEF_ATTR_COLUMN][i])),
                                   int(float(region[DEF_ATTR_DEPTH][i]))])
            
            # append the temporary list to the main list
            #
            counter += num_reg
            coordinates.append(temp_coord)

        # go through the label list and replace null with "null"
        # since Python convert null to NaN automatically
        #
        text = [DEF_LABEL_NULL if pandas.isna(i) else i for i in text]
        
        # get the index, region_id, and confidence value from the dictionary
        #
        index = list(set(region[DEF_ATTR_INDEX]))
        region_id = list(set(region[DEF_ATTR_RINDEX]))
        confidence = list(set(region[DEF_LABEL_CONFIDENCE]))

        # append the information to the dict
        #
        for i, ind in enumerate(index):

            data[ind] = {}

            # append the information to the dictionary
            #
            data[ind][DEF_LABEL_REGION_ID] = str(region_id[i])
            data[ind][DEF_LABEL_TEXT] = text[i]
            data[ind][DEF_LABEL_COORDS] = coordinates[i]
            data[ind][DEF_LABEL_CONFIDENCE] = str(confidence[0])
            data[ind][DEF_LABEL_TISSUE_VALUE] = tissue_type[i]

        # return the dictionaries
        #
        return (header, data)
    
    # method Csv::print_events_from_file
    #
    # arguments:
    #  fname: file name
    #  fp: file pointer
    # 
    # return: a boolean indicating status
    #
    # This method pretty prints the CSV annotation information.
    #
    def print_events_from_file(self, fname, fp = sys.stdout):

        # declare local variables
        #
        _, data = self.read(fname)
        header = []

        # open a file
        #
        if dbgl > ndt.BRIEF:
            fp.write("%s (line: %s) %s: opening (%s)\n" %
                     (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

        file = open(fname, nft.MODE_READ_TEXT)
        if file is None:
            print("Error: %s (line: %s) %s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False

        # get the header of the file using regular expression
        #
        for line in file:
            if re.match(DEF_REGEX_COMMENT, line):
                header.append(
                    line.strip(nft.DELIM_COMMENT).strip(nft.DELIM_NEWLINE))
        
        # close the file and reset the pointer
        #
        if dbgl > ndt.BRIEF:
            fp.write("%s (line: %s) %s: closing (%s)\n" %
                     (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
        file.close()

        # print the header
        #
        for i, header_line in enumerate(header):
            if i < len(header):
                fp.write("%s\n" % (header_line))

        # pretty print the information
        #
        for val in data.values():
            
            # we first create a list of tuples of (row, column) values, and
            # then we use the zip function to join all row and column into
            # their respective lists by unpacking the tuple value with the
            # * operator.
            #
            column_vertices, row_vertices = zip(
                *[(coord[0], coord[1]) for coord in val[DEF_LABEL_COORDS]]
            )

            fp.write(DEF_FMT_REGION % (val[DEF_LABEL_REGION_ID]))

            fp.write(DEF_FMT_ITEM %
                     (DEF_ATTR_TISSUE_VALUE, val[DEF_LABEL_TISSUE_VALUE]))

            fp.write(DEF_FMT_ITEM % (DEF_ATTR_TEXT, val[DEF_LABEL_TEXT]))

            fp.write(DEF_FMT_ITEM % 
                    (DEF_ATTR_CONFIDENCE, val[DEF_LABEL_CONFIDENCE]))

            fp.write(DEF_FMT_VERTS % (DEF_ATTR_VERTICES, min(row_vertices), 
                    max(row_vertices), min(column_vertices), 
                    max(column_vertices)))

        # exit gracefully
        #
        return True

    # method Csv::write_header
    #
    # arguments:
    #  fname: name of input file
    #  fp: pointer to output, default is sys.stdout
    #
    # return: a boolean value representing status
    #
    # This function will get the header dictionary for a CSV
    # file and write the header to fp
    #
    def write_header(self, fname, fp = sys.stdout):

        # extract the basename without an extension
        #
        try :
            bname = path(fp).stem
        except TypeError:
            bname = "sys.stdout"
        
        # check if there's a file pointer
        #
        if fp is not sys.stdout:
            try: 
                fp = open(fp, "w")
            except:
                print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "Unable to find file", fname))
                sys.exit(os.EX_SOFTWARE)

        # read the header and data dictionaries
        #
        header, data = read(fname)

        # check if data or header is empty
        #
        if not header or not data:
            print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "Unable to Read XML file", fname))
            sys.exit(os.EX_SOFTWARE)

        # display debug information
        #
        if dbgl > ndt.BRIEF:
            pprint.pprint("%s (line: %s) %s: printing header API (%s)" %
                          (__FILE__, ndt.__LINE__, ndt.__NAME__, header))
            pprint.pprint("%s (line: %s) %s: printing data API (%s)" %
                          (__FILE__, ndt.__LINE__, ndt.__NAME__, data))

        # check if a file poin
        #
                
        # write the proper CSV header format
        #
        fp.write("%s" % (DEF_CSV_VERSION) + nft.DELIM_NEWLINE)
        fp.write(nft.DELIM_COMMENT + " %s = %s" %
                 (DEF_ATTR_MICRONS,
                  header[DEF_ATTR_MICRONS]) + nft.DELIM_NEWLINE)

        fp.write(nft.DELIM_COMMENT + " %s = %s" %
                 (DEF_ATTR_BNAME, bname) + nft.DELIM_NEWLINE)

        fp.write(nft.DELIM_COMMENT + "%s" %
                 (DEF_FMT_DIMEN %
                  (header[DEF_ATTR_WIDTH], header[DEF_ATTR_HEIGHT])))

        # list of all different tissue types
        #
        tissue_list = []

        # create a list of all unique tissues in data dictionary
        #
        for i in range(len(data)):
            if data[i][DEF_LABEL_TISSUE_VALUE] not in tissue_list:
                tissue_list.append(data[i][DEF_LABEL_TISSUE_VALUE])

        fp.write(nft.DELIM_COMMENT + " %s = " % (DEF_LABEL_TISSUE))
        fp.write(', '.join(tissue_list) + nft.DELIM_NEWLINE)
        fp.write(nft.DELIM_COMMENT + nft.DELIM_NEWLINE)
        fp.write("%s" % (DEF_FMT_HEADER) + nft.DELIM_NEWLINE)

        if fp is not sys.stdout:
            fp.close()

        # exit gracefully
        #
        return True

    # method Csv::write_data
    #
    # arguments:
    #  fname: filename of the input XML file
    #  fp: pointer to output, default is sys.stdout
    #
    # return: a boolean value representing status
    #
    # This function will get the data dictionary containing
    # CSV data information and write it to fp
    #
    def write_data(self, fname, fp = sys.stdout):

        # check if there's a file pointer
        #
        if fp is not sys.stdout:
            try: 
                fp = open(fp, "a")
            except:
                print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "Unable to find file", fname))
                sys.exit(os.EX_SOFTWARE)
    
        # read the header and data dictionaries
        #
        header, data = read(fname)

        # check if data or header is empty
        #
        if not header or not data:
            print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "Unable to Read XML file", fname))
            sys.exit(os.EX_SOFTWARE)

        # display debug information
        #
        if dbgl > ndt.BRIEF:
            pprint.pprint("%s (line: %s) %s: printing header API (%s)" %
                          (__FILE__, ndt.__LINE__, ndt.__NAME__, header))
            pprint.pprint("%s (line: %s) %s: printing data API (%s)" %
                          (__FILE__, ndt.__LINE__, ndt.__NAME__, data))
    
        # loop over the data items
        #
        for index, region_info in data.items():
            for i in range(len(region_info[DEF_LABEL_COORDS])):
                fp.write("%s," % (index))
                fp.write("%s," % (region_info[DEF_LABEL_REGION_ID]))
                fp.write("%s," % (region_info[DEF_LABEL_TISSUE_VALUE]))
                fp.write("%s," % (region_info[DEF_LABEL_TEXT]))
                fp.write("%s," % (i))
                fp.write("%s," % (region_info[DEF_LABEL_COORDS][i][1]))
                fp.write("%s," % (region_info[DEF_LABEL_COORDS][i][0]))
                fp.write("%s," % (region_info[DEF_LABEL_COORDS][i][2]))
                fp.write("%s"  % (region_info[DEF_LABEL_CONFIDENCE] +
                                  nft.DELIM_NEWLINE))
        
        if fp is not sys.stdout:
            fp.close()

        # exit gracefully
        #
        return True

#
# end of class

#
# end of file
