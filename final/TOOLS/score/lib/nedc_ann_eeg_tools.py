#!/usr/bin/env python
#
# file: $NEDC_NFC/class/python/nedc_ann_tools/nedc_ann_tools.py
#
# revision history:
#
# 20220307 (PM): configured the CSV Class to support the new CSV format
# 20210201 (TC): added XML and CSV
# 20200610 (LV): refactored code
# 20200607 (JP): refactored code
# 20170728 (JP): added compare_durations and load_annotations
# 20170716 (JP): upgraded to use the new annotation tools
# 20170714 (NC): created new class structure
# 20170709 (JP): refactored the code
# 20170612 (NC): added parsing and displaying methods
# 20170610 (JP): initial version
#
# This class contains a collection of methods that provide
# the infrastructure for processing annotation-related data.
# ------------------------------------------------------------------------------

# import reqired system modules
#
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from lxml import etree
from xml.dom import minidom

# import required NEDC modules
#
import nedc_debug_tools as ndt
import nedc_file_tools as nft

# ------------------------------------------------------------------------------
#
# global variables are listed here
#
# ------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define a data structure that encapsulates all file types:
#  we use this data structure to access lower-level objects. the key
#  is the type name, the first value is the magic sequence that should
#  appear in the file and the second value is the name of class data member
#  that is used to dynamically bind the subclass based on the type.
#
#  we also need a list of supported versions for utilities to use.
#
FTYPE_LBL = 'lbl_v1.0.0'
FTYPE_TSE = 'tse_v1.0.0'
FTYPE_CSV = 'csv_v1.0.0'
FTYPE_XML = '1.0'

FTYPES = {'lbl': [FTYPE_LBL, 'lbl_d'], 'tse': [FTYPE_TSE, 'tse_d'],
          'csv': [FTYPE_CSV, 'csv_d'], 'xml': [FTYPE_XML, 'xml_d']}
VERSIONS = [FTYPE_LBL, FTYPE_TSE, FTYPE_CSV, FTYPE_XML]

# define numeric constants
#
DEF_CHANNEL = int(-1)

# define the string to check the files" header
#
DEF_CSV_HEADER = "# version = csv_v1.0.0"
DEF_CSV_LABELS = "channel,start_time,stop_time,label,confidence"

# ---
# define constants associated with the Annotation class
#

# ---
# define constants associated with the Lbl class
#

# define a default montage file
#
DEFAULT_MAP_FNAME = "$NEDC_NFC/lib/default_map.txt"

DEFAULT_MONTAGE_FNAME = "$NEDC_NFC/lib/nedc_eas_default_montage.txt"

DEFAULT_XML_CONSTANT_FNAME = "$NEDC_NFC/lib/default_xml_constant.txt"

# define symbols that appear as keys in an lbl file
#
DELIM_LBL_MONTAGE = 'montage'
DELIM_LBL_NUM_LEVELS = 'number_of_levels'
DELIM_LBL_LEVEL = 'level'
DELIM_LBL_SYMBOL = 'symbols'
DELIM_LBL_LABEL = 'label'

# define a list of characters we need to parse out
#
REM_CHARS = [nft.DELIM_BOPEN, nft.DELIM_BCLOSE, nft.DELIM_NEWLINE,
             nft.DELIM_SPACE, nft.DELIM_QUOTE, nft.DELIM_SEMI,
             nft.DELIM_SQUOTE]

# ---
# define constants associated with the Xml class
#

# define the location of the schema file:
#  note that this is version specific since the schema file will evolve
#  over time.
#
DEF_SCHEMA_FILE = "$NEDC_NFC/lib/nedc_eeg_xml_schema_v00.xsd"

# define the root label
#
DEF_XML_EVENTS = "xml_events"
DEF_ROOT = "label"
DEF_ENDPTS = "endpoints"
DEF_TERM = ["term", "tcp_ar"]
DEF_PROB = "probability"

# define the attributes
#
DEF_NAME = "name"
DEF_TYPE = "dtype"

# define tags in template file which their text their text be checked
#
TMPL_ROOT = 'tag'
TMPL_ATTR = 'attrib'
TMPL_VALU = 'value'
TMPL_TEXT = 'text'
TMPL_DTYP = 'dtype'
TMPL_SUBT = 'subtags'
TMPL_OPTL = 'optional'

# define types check
#
PARENT_TYPE = 'parent'
EVENT_TYPE = 'event'
LIST_TYPE = 'list'
DICT_TYPE = 'dict'

# ------------------------------------------------------------------------------
#
# functions listed here
#
# ------------------------------------------------------------------------------

# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()

# function: get_unique_events
#
# arguments:
#  events: events to aggregate
#
# return: a list of unique events
#
# This method combines events if they are of the same start/stop times.
#
def get_unique_events(events):

    # list to store unique events
    #
    unique_events = []

    # make sure events_a are sorted
    #
    events = sorted(events, key=lambda x: (x[0], x[1]))

    # loop until we have checked all events_a
    #
    while len(events) != 0:

        # reset flag
        #
        is_unique = True
        n_start = int(-1)
        n_stop = int(-1)

        # get this event's start/stop times
        #
        start = events[0][0]
        stop = events[0][1]

        # if we are not at the last event
        #
        if len(events) != 1:

            # get next event's start/stop times
            #
            n_start = events[1][0]
            n_stop = events[1][1]

        # if this event's start/stop times are the same as the next event's,
        #  (only do this if we are not at the last event)
        #
        if (n_start == start) and (n_stop == stop) and (len(events) != 1):

            # combine this event's dict with the next event's symbol dict
            #
            for symb in events[1][2]:

                # if the symb is not found in this event's dict
                #
                if symb not in events[0][2]:

                    # add symb to this event's dict
                    #
                    events[0][2][symb] = events[1][2][symb]

                # else if the symb in the next event has a higher prob
                #
                elif events[1][2][symb] > events[0][2][symb]:

                    # update this event's symb with prob from the next event
                    #
                    events[0][2][symb] = events[1][2][symb]
                #
                # end of if/elif
            #
            # end of for

            # delete the next event, it is not unique
            #
            del events[1]
        #
        # end of if

        # loop over unique events
        #
        for unique in unique_events:

            # if the start/stop times of this event is found in unique events
            #
            if (start == unique[0]) and (stop == unique[1]):

                # combine unique event's dict with this event's dict:
                #  iterate over symbs in this event's dict
                #
                for symb in events[0][2]:

                    # if the symb is not found in the unique event's dict
                    #
                    if symb not in unique[2]:

                        # add symb to the unique event's dict
                        #
                        unique[2][symb] = events[0][2][symb]

                    # else if the symb in this event has a higher prob
                    #
                    elif events[0][2][symb] > unique[2][symb]:

                        # update unique event's symb with prob from this event
                        #
                        unique[2][symb] = events[0][2][symb]
                    #
                    # end of if/elif
                #
                # end of for

                # delete this event, it is not unique
                #
                del events[0]
                is_unique = False
                break
            #
            # end of if
        #
        # end of for

        # if this event is still unique
        #
        if is_unique is True:

            # add this event to the unique events
            #
            unique_events.append(events[0])

            # delete this event, it is now stored as unique
            #
            del events[0]
        #
        # end of if
    #
    # end of while

    # exit gracefully
    #
    return unique_events
#
# end of function

# function: compare_durations
#
# arguments:
#  l1: the first list of files
#  l2: the second list of files
#
# return: a boolean value indicating status
#
# This method goes through two lists of files and compares the durations
# of the annotations. If they don't match, it returns false.
#
def compare_durations(l1, l2):

    # display an informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: comparing durations of annotations" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # create an annotation object
    #
    ann = AnnEeg()

    # check the length of the lists
    #
    if len(l1) != len(l2):
        return False

    # loop over the lists together
    #
    for l1_i, l2_i in zip(l1, l2):

        # load the annotations for l1
        #
        if ann.load(l1_i) == False:
            print("Error: %s (line: %s) %s: error loading annotation (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, l1_i))
            return False

        # get the events for l1
        #
        events_l1 = ann.get()
        if events_l1 == None:
            print("Error: %s (line: %s) %s: error getting annotation ((%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, l1_i))
            return False

        # load the annotations for l2
        #
        if ann.load(l2_i) == False:
            print("Error: %s (line: %s) %s: error loading annotation (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, l2_i))
            return False

        # get the events for l2
        #
        events_l2 = ann.get()
        if events_l2 == None:
            print("Error: %s (line: %s) %s: error getting annotation: (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, l2_i))
            return False

        # check the durations
        #
        if round(events_l1[-1][1], ndt.MAX_PRECISION) != \
           round(events_l2[-1][1], ndt.MAX_PRECISION):
            print("Error: %s (line: %s) %s: durations do not match" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            print("\t%s (%f)" % (l1_i, events_l1[-1][1]))
            print("\t%s (%f)" % (l2_i, events_l2[-1][1]))
            return False

    # exit gracefully
    #
    return True
#
# end of function

# function: load_annotations
#
# arguments:
#  list: a list of filenames
#
# return: a list of lists containing all the annotations
#
# This method loops through a list and collects all the annotations.
#
def load_annotations(flist, level=int(0), sublevel=int(0),
                     channel=DEF_CHANNEL):

    # display an informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: loading annotations" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # create an annotation object
    #
    events = []
    ann = AnnEeg()

    # loop over the list
    #
    for fname in flist:

        # load the annotations
        #
        if ann.load(fname) == False:
            print("Error: %s (line: %s) %s: loading annotation for file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return None

        # get the events
        #
        events_tmp = ann.get(level, sublevel, channel)
        if events_tmp == None:
            print("Error: %s (line: %s) %s: error getting annotation (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return None
        events.append(events_tmp)

    # exit gracefully
    #
    return events
#
# end of function

# ------------------------------------------------------------------------------
#
# classes are listed here:
#  there are four classes in this file arranged in this hierarchy
#   AnnEeg -> {Tse, Lbl, Xml, Csv} -> AnnGrEeg
#
# ------------------------------------------------------------------------------

# class: AnnGrEeg
#
# This class implements the main data structure used to hold an annotation.
#
class AnnGrEeg:

    # method: AnnGrEeg::constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self):

        # set the class name
        #
        AnnGrEeg.__CLASS_NAME__ = self.__class__.__name__

        # declare a data structure to hold a graph
        #
        self.graph_d = {}
    #
    # end of method

    # method: AnnGrEeg::create
    #
    # arguments:
    #  lev: level of annotation
    #  sub: sublevel of annotation
    #  chan: channel of annotation
    #  start: start time of annotation
    #  stop: stop time of annotation
    #  symbols: dict of symbols/probabilities
    #
    # return: a boolean value indicating status
    #
    # This method create an annotation in the AG data structure
    #
    def create(self, lev, sub, chan, start, stop, symbols):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "creating annotation in AG data structure"))

        # try to access sublevel dict at level
        #

        try:
            self.graph_d[lev]

            # try to access channel dict at level/sublevel
            #
            try:
                self.graph_d[lev][sub]

                # try to append values to channel key in dict
                #
                try:
                    self.graph_d[lev][sub][chan].append([start, stop, symbols])

                # if appending values failed, finish data structure
                #
                except:

                    # create empty list at chan key
                    #
                    self.graph_d[lev][sub][chan] = []

                    # append values
                    #
                    self.graph_d[lev][sub][chan].append([start, stop, symbols])

            # if accessing channel dict failed, finish data structure
            #
            except:

                # create dict at level/sublevel
                #
                self.graph_d[lev][sub] = {}

                # create empty list at chan
                #
                self.graph_d[lev][sub][chan] = []

                # append values
                #
                self.graph_d[lev][sub][chan].append([start, stop, symbols])

        # if accessing sublevel failed, finish data structure
        #
        except:

            # create dict at level
            #
            self.graph_d[lev] = {}

            # create dict at level/sublevel
            #
            self.graph_d[lev][sub] = {}

            # create empty list at level/sublevel/channel
            #
            self.graph_d[lev][sub][chan] = []

            # append values
            #
            self.graph_d[lev][sub][chan].append([start, stop, symbols])

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: AnnGrEeg::get
    #
    # arguments:
    #  level: level of annotations
    #  sublevel: sublevel of annotations
    #
    # return: events by channel at level/sublevel
    #
    # This method returns the events stored at the level/sublevel argument
    #
    def get(self, level, sublevel, channel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: getting events stored at level/sublevel" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # declare local variables
        #
        events = []

        # try to access graph at level/sublevel/channel
        #
        try:
            events = self.graph_d[level][sublevel][channel]

            # exit gracefully
            #
            return events

        # exit (un)gracefully: if failed, return False
        #
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d/%d)" %
                  (__FILE__, ndt.__LINE__, AnnGrEeg.__CLASS_NAME__,
                   ndt.__NAME__, "level/sublevel/channel not found",
                   level, sublevel, channel))
            return False
    #
    # end of method

    # method: AnnGrEeg::sort
    #
    # arguments: none
    #
    # return: a boolean value indicating status
    #
    # This method sorts annotations by level, sublevel,
    # channel, start, and stop times
    #
    def sort(self):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: %s %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "sorting annotations by",
                   "level, sublevel, channel, start and stop times"))

        # sort each level key by min value
        #
        self.graph_d = dict(sorted(self.graph_d.items()))

        # iterate over levels
        #
        for lev in self.graph_d:

            # sort each sublevel key by min value
            #
            self.graph_d[lev] = dict(sorted(self.graph_d[lev].items()))

            # iterate over sublevels
            #
            for sub in self.graph_d[lev]:

                # sort each channel key by min value
                #
                self.graph_d[lev][sub] = \
                    dict(sorted(self.graph_d[lev][sub].items()))

                # iterate over channels
                #
                for chan in self.graph_d[lev][sub]:

                    # sort each list of labels by start and stop times
                    #
                    self.graph_d[lev][sub][chan] = \
                        sorted(self.graph_d[lev][sub][chan],
                               key=lambda x: (x[0], x[1]))

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: AnnGrEeg::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur, sym, level, sublevel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: adding events of type sym" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # try to access level/sublevel
        #
        try:
            self.graph_d[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, AnnGrEeg.__CLASS_NAME__,
                   ndt.__NAME__, "level/sublevel not found", level, sublevel))
            return False

        # variable to store what time in the file we are at
        #
        mark = 0.0

        # make sure events are sorted
        #
        self.sort()

        # iterate over channels at level/sublevel
        #
        for chan in self.graph_d[level][sublevel]:

            # reset list to store events
            #
            events = []

            # iterate over events at each channel
            #
            for event in self.graph_d[level][sublevel][chan]:

                # ignore if the start or stop time is past the duration
                #
                if (event[0] > dur) or (event[1] > dur):
                    pass

                # ignore if the start time is bigger than the stop time
                #
                elif event[0] > event[1]:
                    pass

                # ignore if the start time equals the stop time
                #
                elif event[0] == event[1]:
                    pass

                # if the beginning of the event is not at the mark
                #
                elif event[0] != mark:

                    # create event from mark->starttime
                    #
                    events.append([mark, event[0], {sym: 1.0}])

                    # add event after mark->starttime
                    #
                    events.append(event)

                    # set mark to the stop time
                    #
                    mark = event[1]

                # if the beginning of the event is at the mark
                #
                else:

                    # store this event
                    #
                    events.append(event)

                    # set mark to the stop time
                    #
                    mark = event[1]
            #
            # end of for

            # after iterating through all events, if mark is not at dur
            #
            if mark != dur:

                # create event from mark->dur
                #
                events.append([mark, dur, {sym: 1.0}])

            # store events as the new events in self.graph_d
            #
            self.graph_d[level][sublevel][chan] = events
        #
        # end of for

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: AnnGrEeg::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym
    #
    def delete(self, sym, level, sublevel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: deleting events of type sym" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # try to access level/sublevel
        #
        try:
            self.graph_d[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, AnnGrEeg.__CLASS_NAME__,
                   ndt.__NAME__, "level/sublevel not found", level, sublevel))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in self.graph_d[level][sublevel]:

            # get events at chan
            #
            events = self.graph_d[level][sublevel][chan]

            # keep only the events that do not contain sym
            #
            events = [e for e in events if sym not in e[2].keys()]

            # store events in self.graph_d
            #
            self.graph_d[level][sublevel][chan] = events
        #
        # end of for

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: AnnGrEeg::get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method returns the entire graph, instead of a
    # level/sublevel/channel.
    #
    def get_graph(self):
        return self.graph_d
    #
    # end of method

    # method: AnnGrEeg::set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph.
    #
    def set_graph(self, graph):
        self.graph_d = graph
        self.sort()
        return True
    #
    # end of method

    # method: AnnGrEeg::delete_graph
    #
    def delete_graph(self):
        self.graph_d  = {}
        return True
#
# end of class

# class: Tse
#
# This class contains methods to manipulate time-synchronous event files.
#
class Tse:

    # method: Tse::constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self):

        # set the class name
        #
        Tse.__CLASS_NAME__ = self.__class__.__name__

        # declare Graph object, to store annotations
        #
        self.graph_d = AnnGrEeg()
    #
    # end of method

    # method: Tse::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: loading annotation from file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # open file
        #
        with open(fname, nft.MODE_READ_TEXT) as fp:

            # loop over lines in file
            #
            for line in fp:

                # clean up the line
                #
                line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                           .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)
                check = line.replace(nft.DELIM_SPACE, nft.DELIM_NULL)

                # throw away commented, blank lines, version lines
                #
                if check.startswith(nft.DELIM_COMMENT) or \
                   check.startswith(nft.DELIM_VERSION) or \
                   len(check) == 0:
                    continue

                # split the line
                #
                val = {}
                parts = line.split()

                try:
                    # loop over every part, starting after start/stop times
                    #
                    for i in range(2, len(parts), 2):

                        # create dict with label as key, prob as value
                        #
                        val[parts[i]] = float(parts[i+1])

                    # create annotation in AG
                    #
                    self.graph_d.create(int(0), int(0), int(-1),
                                        float(parts[0]), float(parts[1]), val)
                except:
                    print("Error: %s (line: %s) %s::%s %s (%s)" %
                          (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__,
                           ndt.__NAME__, "invalid annotation", line))
                    return False

        # make sure graph is sorted after loading
        #
        self.graph_d.sort()

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Tse::get
    #
    # arguments:
    #  level: level of annotations to get
    #  sublevel: sublevel of annotations to get
    #
    # return: events at level/sublevel by channel
    #
    # This method gets the annotations stored in the AG at level/sublevel.
    #
    def get(self, level, sublevel, channel):
        events = self.graph_d.get(level, sublevel, channel)
        return events
    #
    # end of method

    # method: Tse::display
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a boolean value indicating status
    #
    # This method displays the events from a flat AG.
    #
    def display(self, level, sublevel, fp=sys.stdout):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: displaying events from flag AG" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__, ndt.__NAME__,
                   "level/sublev not in graph", level, sublevel))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in graph[level][sublevel]:

            # iterate over events for each channel
            #
            for event in graph[level][sublevel][chan]:
                start = event[0]
                stop = event[1]

                # create a string with all symb/prob pairs
                #
                pstr = ""
                for symb in event[2]:
                    pstr += " %8s %10.4f" % (symb, event[2][symb])

                # display event
                #
                fp.write("%10s: %10.4f %10.4f%s\n" %
                         ('ALL', start, stop, pstr))

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Tse::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method writes the events to a .tse file
    #
    def write(self, ofile, level, sublevel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: writing events to .tse file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # make sure graph is sorted
        #
        self.graph_d.sort()

        # get graph
        #
        graph = self.get_graph()

        # try to access the graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__,
                   ndt.__NAME__, "level/sublevel not in graph",
                   level, sublevel))
            return False

        # list to collect all events
        #
        events = []

        # iterate over channels at level/sublevel
        #
        for chan in graph[level][sublevel]:

            # iterate over events for each channel
            #
            for event in graph[level][sublevel][chan]:

                # store every channel's events in one list
                #
                events.append(event)

        # remove any events that are not unique
        #
        events = get_unique_events(events)

        # open file with write
        #
        with open(ofile, nft.MODE_WRITE_TEXT) as fp:

            # write version
            #
            fp.write("version = %s\n" % FTYPES['tse'][0])
            fp.write(nft.DELIM_NEWLINE)

            # iterate over events
            #
            for event in events:

                # create symb/prob string from dict
                #
                pstr = ""
                for symb in event[2]:
                    pstr += " %s %.4f" % (symb, event[2][symb])

                # write event
                #
                fp.write("%.4f %.4f%s\n" % (event[0], event[1], pstr))

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Tse::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur, sym, level, sublevel):
        return self.graph_d.add(dur, sym, level, sublevel)
    #
    # end of method

    # method: Tse::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym.
    #
    def delete(self, sym, level, sublevel):
        return self.graph_d.delete(sym, level, sublevel)
    #
    # end of method

    # method: Tse::get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()
    #
    # end of method

    # method: Tse::set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph.
    #
    def set_graph(self, graph):
        return self.graph_d.set_graph(graph)
    #
    # end of method

#
# end of class

# class: Lbl
#
# This class implements methods to manipulate label files.
#
class Lbl:

    # method: Lbl::constructor
    #
    # arguments: none
    #
    # return: none
    #
    # This method constructs Ag
    #
    def __init__(self):

        # set the class name
        #
        Lbl.__CLASS_NAME__ = self.__class__.__name__

        # declare variables to store info parsed from lbl file
        #
        self.chan_map_d = {int(-1): 'all'}
        self.montage_lines_d = []
        self.symbol_map_d = {}
        self.num_levels_d = int(1)
        self.num_sublevels_d = {int(0): int(1)}

        # declare AG object to store annotations
        #
        self.graph_d = AnnGrEeg()
    #
    # end of method

    # method: Lbl::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: loading annotation from file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # open file
        #
        fp = open(fname, nft.MODE_READ_TEXT)

        # loop over lines in file
        #
        for line in fp:

            # clean up the line
            #
            line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                       .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)

            # parse a single montage definition
            #
            if line.startswith(DELIM_LBL_MONTAGE):
                try:
                    chan_num, name, montage_line = \
                        self.parse_montage(line)
                    self.chan_map_d[chan_num] = name
                    self.montage_lines_d.append(montage_line)
                except:
                    print("Error: %s (line: %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing montage", line))
                    fp.close()
                    return False

            # parse the number of levels
            #
            elif line.startswith(DELIM_LBL_NUM_LEVELS):
                try:
                    self.num_levels_d = self.parse_numlevels(line)
                except:
                    print("Error: %s (line: %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing number of levels",
                           line))
                    fp.close()
                    return False

            # parse the number of sublevels at a level
            #
            elif line.startswith(DELIM_LBL_LEVEL):
                try:
                    level, sublevels = self.parse_numsublevels(line)
                    self.num_sublevels_d[level] = sublevels

                except:
                    print("Error: %s (line: %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing num of sublevels",
                           line))
                    fp.close()
                    return False

            # parse symbol definitions at a level
            #
            elif line.startswith(DELIM_LBL_SYMBOL):
                try:
                    level, mapping = self.parse_symboldef(line)
                    self.symbol_map_d[level] = mapping
                except:
                    print("Error: %s (line %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing symbols", line))
                    fp.close()
                    return False

            # parse a single label
            #
            elif line.startswith(DELIM_LBL_LABEL):
                lev, sub, start, stop, chan, symbols = \
                    self.parse_label(line)
                try:
                    lev, sub, start, stop, chan, symbols = \
                        self.parse_label(line)
                except:
                    print("Error: %s (line %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing label", line))
                    fp.close()
                    return False

                # create annotation in AG
                #
                status = self.graph_d.create(lev, sub, chan,
                                             start, stop, symbols)

        # close file
        #
        fp.close()

        # sort labels after loading
        #
        self.graph_d.sort()

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: Lbl::get
    #
    # arguments:
    #  level: level value
    #  sublevel: sublevel value
    #
    # return: events by channel from AnnGrEeg
    #
    # This method returns the events at level/sublevel
    #
    def get(self, level, sublevel, channel):

        # get events from AG
        #
        events = self.graph_d.get(level, sublevel, channel)

        # exit gracefully
        #
        return events
    #
    # end of method

    # method: Lbl::display
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a boolean value indicating status
    #
    # This method displays the events from a flat AG
    #
    def display(self, level, sublevel, fp=sys.stdout):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: displaying events from flat AG" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get graph
        #
        graph = self.get_graph()

        # try to access level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            sys.stdout.write("Error: %s (line: %s) %s::%s: %s (%d/%d)" %
                             (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                              ndt.__NAME__, "level/sublevel not found",
                              level, sublevel))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in graph[level][sublevel]:

            # iterate over events at chan
            #
            for event in graph[level][sublevel][chan]:

                # find max probability
                #
                max_prob = max(event[2].values())

                # iterate over symbols in dictionary
                #
                for symb in event[2]:

                    # if the value of the symb equals the max prob
                    #
                    if event[2][symb] == max_prob:

                        # set max symb to this symbol
                        #
                        max_symb = symb
                        break

                # display event
                #
                fp.write("%10s: %10.4f %10.4f %8s %10.4f\n" %
                         (self.chan_map_d[chan], event[0], event[1],
                          max_symb, max_prob))

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Lbl::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method writes events to a .lbl file.
    #
    def write(self, ofile, level, sublevel):

        # make sure graph is sorted
        #
        self.graph_d.sort()

        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s: %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "level/sublevel not found", level, sublevel))
            return False

        # open file with write
        #
        with open(ofile, nft.MODE_WRITE_TEXT) as fp:

            # write version
            #
            fp.write(nft.DELIM_NEWLINE)
            fp.write("version = %s\n" % FTYPES['lbl'][0])
            fp.write(nft.DELIM_NEWLINE)

            # if montage_lines is blank, we are converting from tse to lbl.
            #
            # create symbol map from tse symbols
            #
            if len(self.montage_lines_d) == 0:

                # variable to store the number of symbols
                #
                num_symbols = 0

                # create a dictionary at level 0 of symbol map
                #
                self.symbol_map_d[int(0)] = {}

                # iterate over all events stored in the 'all' channels
                #
                for event in graph[level][sublevel][int(-1)]:

                    # iterate over symbols in each event
                    #
                    for symbol in event[2]:

                        # if the symbol is not in the symbol map
                        #
                        if symbol not in self.symbol_map_d[0].values():

                            # map num_symbols interger to symbol
                            #
                            self.symbol_map_d[0][num_symbols] = symbol

                            # increment num_symbols
                            #
                            num_symbols += 1

            # write montage lines
            #
            for line in self.montage_lines_d:
                fp.write("%s\n" % line)

            fp.write(nft.DELIM_NEWLINE)

            # write number of levels
            #
            fp.write("number_of_levels = %d\n" % self.num_levels_d)
            fp.write(nft.DELIM_NEWLINE)

            # write number of sublevels
            #
            for lev in self.num_sublevels_d:
                fp.write("level[%d] = %d\n" %
                         (lev, self.num_sublevels_d[lev]))
            fp.write(nft.DELIM_NEWLINE)

            # write symbol definitions
            #
            for lev in self.symbol_map_d:
                fp.write("symbols[%d] = %s\n" %
                         (lev, str(self.symbol_map_d[lev])))
            fp.write(nft.DELIM_NEWLINE)

            # iterate over channels at level/sublevel
            #
            for chan in graph[level][sublevel]:

                # iterate over events in chan
                #
                for event in graph[level][sublevel][chan]:

                    # create string for probabilities
                    #
                    pstr = "["

                    # iterate over symbol map
                    #
                    for symb in self.symbol_map_d[level].values():

                        # if the symbol is found in the event
                        #
                        if symb in event[2]:
                            pstr += (str(event[2][symb]) + nft.DELIM_COMMA +
                                     nft.DELIM_SPACE)
                        else:
                            pstr += '0.0' + nft.DELIM_COMMA + nft.DELIM_SPACE

                    # remove the ', ' from the end of pstr
                    #
                    pstr = pstr[:len(pstr) - 2] + "]}"

                    # write event
                    #
                    fp.write("label = {%d, %d, %.4f, %.4f, %s, %s\n" %
                             (level, sublevel, event[0], event[1], chan, pstr))
        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Lbl::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym
    #
    def add(self, dur, sym, level, sublevel):
        return self.graph_d.add(dur, sym, level, sublevel)

    # method: Lbl::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym
    #
    def delete(self, sym, level, sublevel):
        return self.graph_d.delete(sym, level, sublevel)

    # method: Lbl::get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()
    #
    # end of method

    # method: Lbl::set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph
    #
    def set_graph(self, graph):
        return self.graph_d.set_graph(graph)
    #
    # end of method

    # method: Lbl::parse_montage
    #
    # arguments:
    #  line: line from label file containing a montage channel definition
    #
    # return:
    #  channel_number: an integer containing the channel map number
    #  channel_name: the channel name corresponding to channel_number
    #  montage_line: entire montage def line read from file
    #
    # This method parses a montage line into it's channel name and number.
    # Splitting a line by two values easily allows us to get an exact
    # value/string from a line of definitions
    #
    def parse_montage(self, line):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: parsing montage by channel name, number" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # split between '=' and ',' to get channel number
        #
        channel_number = int(
            line.split(nft.DELIM_EQUAL)[1].split(nft.DELIM_COMMA)[0].strip())

        # split between ',' and ':' to get channel name
        #
        channel_name = line.split(
            nft.DELIM_COMMA)[1].split(nft.DELIM_COLON)[0].strip()

        # remove chars from montage line
        #
        montage_line = line.strip().strip(nft.DELIM_NEWLINE)

        # exit gracefully
        #
        return [channel_number, channel_name, montage_line]
    #
    # end of method

    # method: Lbl::parse_numlevels
    #
    # arguments:
    #  line: line from label file containing the number of levels
    #
    # return: an integer containing the number of levels defined in the file
    #
    # This method parses the number of levels in a file.
    #
    def parse_numlevels(self, line):

        # split by '=' and remove extra characters
        #
        return int(line.split(nft.DELIM_EQUAL)[1].strip())
    #
    # end of method

    # method: Lbl::parse_numsublevels
    #
    # arguments:
    #  line: line from label file containing number of sublevels in level
    #
    # return:
    #  level: level from which amount of sublevels are contained
    #  sublevels: amount of sublevels in particular level
    #
    # This method parses the number of sublevels per level in the file
    #
    def parse_numsublevels(self, line):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: parsing number of sublevels per level" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # split between '[' and ']' to get level
        #
        level = int(line.split(
            nft.DELIM_OPEN)[1].split(nft.DELIM_CLOSE)[0].strip())

        # split by '=' and remove extra characters
        #
        sublevels = int(line.split(nft.DELIM_EQUAL)[1].strip())

        # exit gracefully
        #
        return [level, sublevels]
    #
    # end of method

    # method: Lbl::parse_symboldef
    #
    # arguments:
    #  line: line from label fiel containing symbol definition for a level
    #
    # return:
    #  level: an integer containing the level of this symbol definition
    #  mappings: a dict containing the mapping of symbols for this level
    #
    # This method parses a symbol definition line into a specific level,
    # the corresponding symbol mapping as a dictionary.
    #
    def parse_symboldef(self, line):

        # split by '[' and ']' to get level of symbol map
        #
        level = int(line.split(nft.DELIM_OPEN)[1].split(nft.DELIM_CLOSE)[0])

        # remove all characters to remove, and split by ','
        #
        syms = ''.join(c for c in line.split(nft.DELIM_EQUAL)[1]
                       if c not in REM_CHARS)

        symbols = syms.split(nft.DELIM_COMMA)

        # create a dict from string, split by ':'
        #   e.g. '0: seiz' -> mappings[0] = 'seiz'
        #
        mappings = {}
        for s in symbols:
            mappings[int(s.split(':')[0])] = s.split(':')[1]

        # exit gracefully
        #
        return [level, mappings]
    #
    # end of method

    # method: Lbl::parse_label
    #
    # arguments:
    #  line: line from label file containing an annotation label
    #
    # return: all information read from .ag file
    #
    # this method parses a label definition into the values found in the label
    #
    def parse_label(self, line):

        # dict to store symbols/probabilities
        #
        symbols = {}

        # remove characters to remove, and split data by ','
        #
        lines = ''.join(c for c in line.split(nft.DELIM_EQUAL)[1]
                        if c not in REM_CHARS)

        data = lines.split(nft.DELIM_COMMA)

        # separate data into specific variables
        #
        level = int(data[0])
        sublevel = int(data[1])
        start = float(data[2])
        stop = float(data[3])

        # the channel value supports either 'all' or channel name
        #
        try:
            channel = int(data[4])
        except:
            channel = int(-1)

        # parse probabilities
        #
        probs = lines.split(
            nft.DELIM_OPEN)[1].strip(nft.DELIM_CLOSE).split(nft.DELIM_COMMA)

        # set every prob in probs to type float
        #
        probs = list(map(float, probs))

        # convert the symbol map values to a list
        #
        map_vals = list(self.symbol_map_d[level].values())

        # iterate over symbols
        #
        for i in range(len(self.symbol_map_d[level].keys())):

            if probs[i] > 0.0:

                # set each symbol equal to the corresponding probability
                #
                symbols[map_vals[i]] = probs[i]

        # exit gracefully
        #
        return [level, sublevel, start, stop, channel, symbols]
    #
    # end of method

    # method: Lbl::update_montage
    #
    # arguments:
    #  montage_file: montage file
    #
    # return: a boolean value indicating status
    #
    # this method updates a montage file to class value
    #
    def update_montage(self, montage_file):

        # update new montage file, if input montage file is None, update
        # with the default montage
        #
        if montage_file is None or montage_file == "None":
            self.montage_fname_d = nft.get_fullpath(DEFAULT_MONTAGE_FNAME)
        else:
            self.montage_fname_d = nft.get_fullpath(montage_file)

        montage_fp = open(nft.get_fullpath(self.montage_fname_d),
                          nft.MODE_READ_TEXT)
        # loop over lines in file
        #
        lines = montage_fp.readlines()
        for line in lines:
            # clean up the line
            #
            line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                       .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)

            # parse a single montage definition
            #
            if line.startswith(DELIM_LBL_MONTAGE):
                try:
                    chan_num, name, montage_line = \
                        self.parse_montage(line)
                    self.chan_map_d[chan_num] = name
                    self.montage_lines_d.append(montage_line)
                except:
                    print("Error: %s (line %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing montage", line))
                    montage_file.close()
                    return False

            # parse symbol definitions at a level
            #
            elif line.startswith(DELIM_LBL_SYMBOL):
                try:
                    level, mapping = self.parse_symboldef(line)
                    self.symbol_map_d[level] = mapping
                except:
                    print("Error: %s (line %s) %s::%s: %s (%s)" %
                          (__FILE__, ndt.__LINE__, Lbl.__CLASS_NAME__,
                           ndt.__NAME__, "error parsing symbols", line))
                    montage_fp.close()
                    return False
#
# end of class

# class: Csv
#
#
class Csv:

    # method: Csv::constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self, map_f):

        # set the class name
        #
        Csv.__CLASS_NAME__ = self.__class__.__name__

        # declare variables to store infor parsed from csv file
        #
        self.chan_map_d = {int(-1): "TERM"}
        self.montage_fname_d = map_f

        if not self.montage_fname_d:
            self.update_montage(None)

        # declare Graph object, to store annotations
        #
        self.graph_d = AnnGrEeg()
    #
    # end of method

    # method: Csv::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: loading annotation from file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get the header
        #
        header = nft.extract_comments(fname)
        
        # get the file duration
        #
        file_durations_tag, file_durations_val = header[3]

        # open file
        #
        with open(fname, nft.MODE_READ_TEXT) as fp:

            # loop over lines in file
            #
            for line in fp:

                # clean up the line
                #
                line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                           .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)
                check = line.replace(nft.DELIM_SPACE, nft.DELIM_NULL)

                # get montage from montage file provided in csv file
                #
                if check.startswith(nft.DELIM_COMMENT + "annotation"):

                    # if user doesn't input map file, use map file described in csv file
                    #
                    if (self.montage_fname_d is None):
                        try:
                            self.montage_fname_d = check.split(
                                nft.DELIM_EQUAL)[1]
                        except:
                            self.montage_fname_d = check.split(
                                nft.DELIM_COLON)[1]

                        # update montage and chan map with montage provided in csv file
                        #
                        try:
                            status = self.update_montage(self.montage_fname_d)
                            if not status:
                                return False
                        except:
                            # if cannot use montage provide in csv file, use DEFAULT montage
                            #
                            self.montage_fname_d = None
                            status = self.update_montage(self.montage_fname_d)

                    # if user provide input map file
                    #
                    else:
                        status = self.update_montage(self.montage_fname_d)

                    if not status:
                        return False

                # throw away commented, blank lines, version lines
                #
                if check.startswith(nft.DELIM_COMMENT) or \
                   check.startswith(nft.DELIM_VERSION) or \
                   len(check) == 0 or \
                        DEF_CSV_LABELS in line:
                    continue

                # check if montage is still None then use default montage
                #
                if self.montage_fname_d is None:
                    status = self.update_montage(None)
                    if not status:
                        return False

                # split the line
                #
                val = {}
                parts = line.split(nft.DELIM_COMMA)

                try:
                    # loop over every part, starting after start/stop times
                    #
                    for i in range(3, len(parts), 3):

                        # create dict with label as key, prob as value
                        #
                        if i < (len(parts) - 1):
                            val[parts[i]] = float(parts[i+1])
                        else:
                            val[parts[i]] = None
                        
                    val[file_durations_tag] = file_durations_val[:-5]
                    
                    # get chan idx
                    #
                    chan = parts[0].replace(nft.DELIM_QUOTE, nft.DELIM_NULL)

                    # add annotation to all channels if there is "TERM"
                    # else find chan index before add to graph
                    #
                    if chan == "TERM":
                        chan = int(-1)
                    else:
                        chan = next(key for key, value
                                    in self.chan_map_d.items()
                                    if value == chan)

                    # create annotation in AG
                    #
                    self.graph_d.create(int(0), int(0), chan,
                                        float(parts[1]), float(parts[2]), val)

                except:
                    print("Error: %s (line: %s) %s::%s %s (%s)" %
                          (__FILE__, ndt.__LINE__, Csv.__CLASS_NAME__,
                           ndt.__NAME__, "invalid annotation", line))
                    return False

        # make sure graph is sorted after loading
        #
        self.graph_d.sort()

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Csv::get
    #
    # arguments:
    #  level: level of annotations to get
    #  sublevel: sublevel of annotations to get
    #
    # return: events at level/sublevel by channel
    #
    # This method gets the annotations stored in the AG at level/sublevel.
    #
    def get(self, level, sublevel, channel):
        events = self.graph_d.get(level, sublevel, channel)
        return events
    #
    # end of method

    # method: Csv::display
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a boolean value indicating status
    #
    # This method displays the events from a flat AG.
    #
    def display(self, level, sublevel, fp=sys.stdout):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: displaying events from flag AG" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__, ndt.__NAME__,
                   "level/sublev not in graph", level, sublevel))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in graph[level][sublevel]:

            # iterate over events for each channel
            #
            for event in graph[level][sublevel][chan]:

                # find max probability
                #
                max_prob = max(event[2].values())

                # iterate over symbols in dictionary
                #
                for symb in event[2]:

                    # if the value of the symb equals the max prob
                    #
                    if event[2][symb] == max_prob:

                        # set max symb to this symbol
                        #
                        max_symb = symb
                        break

                # display event
                #
                if max_prob is not None:
                    fp.write("%10s: %10.4f %10.4f %8s %10.4f\n" %
                             (self.chan_map_d[chan], event[0], event[1],
                              max_symb, max_prob))
                else:
                    fp.write("%10s: %10.4f %10.4f %8s\n" %
                             (self.chan_map_d[chan], event[0], event[1],
                              max_symb))
        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Csv::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method writes the events to a .tse file
    #
    def write(self, ofile, level, sublevel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: writing events to .csv file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # make sure graph is sorted
        #
        self.graph_d.sort()

        # get graph
        #
        graph = self.get_graph()

        # try to access the graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Tse.__CLASS_NAME__,
                   ndt.__NAME__, "level/sublevel not in graph",
                   level, sublevel))
            return False

        # list to collect all events
        #
        events = []

        # iterate over channels at level/sublevel
        #
        for chan in graph[level][sublevel]:

            # iterate over events for each channel
            #
            for event in graph[level][sublevel][chan]:

                # store every channel's events in one list
                #
                events.append((chan, event))

        # remove any events that are not unique
        #
        events = get_unique_events(events)

        # get duration 
        #
        file_start_time, file_end_time = float("inf"), float("-inf")

        # get the durations, end points and channels
        #
        for channel_index, data in graph[0][0].items():

            for start, stop, _ in data:
                file_start_time = min(file_start_time, start)
                file_end_time = max(file_end_time, stop)


        # open file with write
        #
        with open(ofile, nft.MODE_WRITE_TEXT, newline="\n") as fp:

            # write version
            #
            fp.write("# version = %s\n" % FTYPES['csv'][0])

            # write the bname
            #
            fp.write("# bname = %s\n" %
                     os.path.splitext(os.path.basename(ofile))[0])

            # write the duration
            #
            # fp.write("# duration = %.4f\n" %
            #          float(file_end_time - file_start_time))

            # write map and labelfile
            #
            fp.write("# map file: %s\n" % self.montage_fname_d)
            fp.write("# annotation label file: %s\n" %
                     nft.get_fullpath(DEFAULT_MAP_FNAME))
            fp.write(nft.DELIM_COMMENT)
            fp.write(nft.DELIM_NEWLINE)
            fp.write("%s\n" % (DEF_CSV_LABELS))

            # iterate over events
            #
            for event in events:

                # create symb/prob string from dict
                #
                pstr = ""
                for symb in event[1][2]:
                    if event[1][2][symb] is not None:
                        pstr += "%s,%.4f" % (symb, event[1][2][symb])
                    else:
                        pstr += "%s" % (symb)

                # write event
                #
                fp.write("%s,%.4f,%.4f,%s\n" %
                         (self.chan_map_d[event[0]],
                          event[1][0], event[1][1], pstr))

        # exit gracefully
        #
        return True

    # method: Csv::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur, sym, level, sublevel):
        return self.graph_d.add(dur, sym, level, sublevel)
    #
    # end of method

    # method: Csv::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym.
    #
    def delete(self, sym, level, sublevel):
        return self.graph_d.delete(sym, level, sublevel)
    #
    # end of method

    # method: Csv::get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()
    #
    # end of method


    # method: Csv::set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph.
    #
    def set_graph(self, graph):
        return self.graph_d.set_graph(graph)
    #
    # end of method

    # method: Csv::parse_montage
    #
    # arguments:
    #  line: line from label file containing a montage channel definition
    #
    # return:
    #  channel_number: an integer containing the channel map number
    #  channel_name: the channel name corresponding to channel_number
    #  montage_line: entire montage def line read from file
    #
    # This method parses a montage line into it's channel name and number.
    # Splitting a line by two values easily allows us to get an exact
    # value/string from a line of definitions
    #
    def parse_montage(self, line):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: parsing montage by channel name, number" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # split between '=' and ',' to get channel number
        #
        channel_number = int(
            line.split(nft.DELIM_EQUAL)[1].split(nft.DELIM_COMMA)[0].strip())

        # split between ',' and ':' to get channel name
        #
        channel_name = line.split(
            nft.DELIM_COMMA)[1].split(nft.DELIM_COLON)[0].strip()

        # remove chars from montage line
        #
        montage_line = line.strip().strip(nft.DELIM_NEWLINE)

        # exit gracefully
        #
        return [channel_number, channel_name, montage_line]
    #
    # end of method

    # method: Csv::update_montage
    #
    # argument:
    #  montage_file: montage file
    #
    # return: a boolean value indicating status
    #
    # This method updates the chan_map_d and montage_fname_b based on
    # the input montage
    #
    def update_montage(self, montage_file):

        # update new montage file, if input montage file is None, update
        # with the default montage
        #
        if montage_file is None or montage_file == "None":
            self.montage_fname_d = nft.get_fullpath(DEFAULT_MONTAGE_FNAME)
        else:
            self.montage_fname_d = nft.get_fullpath(montage_file)

        montage_fp = open(self.montage_fname_d,
                          nft.MODE_READ_TEXT)
        try:
            self.chan_map_d = {int(-1): "TERM"}
            for line_mont in montage_fp:
                if line_mont.startswith(DELIM_LBL_MONTAGE):
                    chan_num, name, montage_line = \
                        self.parse_montage(line_mont)
                    self.chan_map_d[chan_num] = name
        except:
            print("Error: %s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, Csv.__CLASS_NAME__,
                   ndt.__NAME__, "error parsing montage file",
                   line_mont))
            montage_fp.close()
            return False
        return True

# class: Xml
#
class Xml:

    # method: Xml::constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self, schema_a, map_f):

        # set the class name
        #
        Xml.__CLASS_NAME__ = self.__class__.__name__

        # declare Graph object, to store annotations
        #
        self.graph_d = AnnGrEeg()

        # if no schema is specified, set it to the default
        #
        if schema_a == None:
            self.schema_fname_d = nft.get_fullpath(DEF_SCHEMA_FILE)
        else:
            self.schema_fname_d = nft.get_fullpath(schema_a)

        # set a variable to hold the processed schema
        #
        self.schema_d = None

        # if no ann map is specified, set it to default
        #
        if map_f == None:
            self.xml_map_fname_d = nft.get_fullpath(DEFAULT_XML_CONSTANT_FNAME)
        else:
            self.xml_map_fname_d = nft.get_fullpath(map_f)

        # declare montage and channel map
        #
        self.chan_map_d = {int(-1): "TERM"}
        self.montage_fname_d = None
        self.update_montage(self.montage_fname_d)

        self.xml_events = None

        self.parse_events()

    #
    # end of method

    # method: Xml::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname):

        # validate based on schema
        #
        status = self.validate(fname)
        if status is not True:
            print("Error: %s (line: %s) %s: invalid xml file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False
        else:
            root = ET.parse(fname).getroot()

            xml_dict = self.tree_to_dict(root)

            # this only sets the annotation to level 0 and sub-level 0
            #
            # TODO: Fix this to support multi-level graph
            #
            graph = {0 : {0: dict(xml_dict)}}

            self.graph_d.graph_d = graph

        # make sure graph is sorted after loading
        #
        self.graph_d.sort()

        return True
    #
    # end of method

    # method: Xml::get
    #
    # arguments:
    #  level: level of annotations to get
    #  sublevel: sublevel of annotations to get
    #
    # return: events at level/sublevel by channel
    #
    # This method gets the annotations stored in the AG at level/sublevel.
    #
    def get(self, level, sublevel, channel):
        events = self.graph_d.get(level, sublevel, channel)
        return events
    #
    # end of method

    # method: Xml::display
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a boolean value indicating status
    #
    # This method displays the events from a flat AG.
    #
    def display(self, level, sublevel, fp=sys.stdout):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: displaying events from flag AG" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level][sublevel]
        except:
            print("Error: %s (line: %s) %s::%s %s (%d/%d)" %
                  (__FILE__, ndt.__LINE__, Xml.__CLASS_NAME__, ndt.__NAME__,
                   "level/sublev not in graph", level, sublevel))
            return False

        for chan in graph[level][sublevel]:
            # iterate over events for each channel
            #
            for event in graph[level][sublevel][chan]:
                start = event[0]
                stop = event[1]

                # create a string with all symb/prob pairs
                #
                pstr = ""
                for symb in event[2]:
                    pstr += " %8s %10.4f" % (symb, event[2][symb])

                if chan != -1:
                    chan_a = chan
                else:
                    chan_a = -1
                # display event
                #
                fp.write("%10s: %10.4f %10.4f%s\n" %
                         (self.chan_map_d[chan_a], start, stop, pstr))

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: Xml::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method writes the events to a .tse file
    #
    def write(self, ofile, level, sublevel):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: writing events to .tse file" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # make sure graph is sorted
        #
        self.graph_d.sort()

        # get graph
        #
        graph = self.get_graph()

        # distribute to all channel if there is chan = -1 in graph
        # this is for case of graph contains csv annotations where both TERM
        # and specific channels are used
        #
        # THIS DON"T WORK FIX THIS 
        for lev in graph:
            for sublev in graph[lev]:

                # check if there are more than 1 channel and one of them is -1
                #
                if (-1 in list(graph[lev][sublev].keys())
                        and len(graph[lev][sublev]) > 1):

                    # assign annotation from channel -1 to every channel
                    #
                    ann = graph[lev][sublev][-1]
                    for chan in self.chan_map_d.keys():
                        if chan not in graph[lev][sublev].keys():
                            graph[lev][sublev].update({chan: ann})
                        else:
                            graph[lev][sublev][chan] = \
                                graph[lev][sublev][chan] + ann

                    # remove chan -1
                    #
                    del graph[lev][sublev][-1]
        
        # Set up a new XML file
        #
        self.set_graph(graph)
        graph = self.get_graph()

        file_start_time, file_end_time = float("inf"), float("-inf")
        channels = set()
        try:
            file_duration = graph[0][0][0][0][-1]["duration"]
            file_duration += " secs"
        except:
            file_duration = "Input file does not have a duration"

        # get the durations, end points and channels
        #
        for channel_index, data in graph[0][0].items():

            channels.add(self.chan_map_d[channel_index])

            for start, stop, _ in data:
                file_start_time = min(file_start_time, start)
                file_end_time = max(file_end_time, stop)

        # set up the root
        #
        root = ET.Element("root")

        # add the bname
        # 
        bname = ET.SubElement(root, "bname")
        bname.text = Path(ofile).stem

        # add the duration
        #
        duration = ET.SubElement(root, "duration")
        duration.text = "%s" % (file_duration)

        # add the montage file
        #
        montage_file = ET.SubElement(root, "montage_file")
        montage_file.text = self.montage_fname_d

        # add the annotation label_file
        #
        annotation_label_file = ET.SubElement(root, "annotation_label_file")
        annotation_label_file.text = nft.get_fullpath(DEFAULT_MAP_FNAME)

        # set up the label
        #
        label = ET.SubElement(root, "label", name= Path(ofile).stem, dtype="parents")

        # add the endpoints 
        #
        endpoints = ET.SubElement(label, "endpoints", name="endpoints", dtype="list")
        endpoints.text = "[%.4f, %.4f]" % (file_start_time, file_end_time)

        # add the montage_channels
        #
        montage_channels = ET.SubElement(label, "montage_channels", name="montage_channels", dtype="parent")
        
        # add all the channels to the xml
        #
        for channel in channels:
            montage_channels.append(ET.Element("channel", name=channel, dtype="*"))

        # writes the start time and end time of each event under the correct channels
        #
        for channel_index, data in graph[0][0].items():

            parent_channel = label.find('montage_channels/channel[@name=\'%s\']' % (self.chan_map_d[channel_index]))

            for start, stop, tag_probability in data:

                event_tag, event_probability = next(iter(tag_probability.items()))

                tag = ET.SubElement(parent_channel, "event", name=str(event_tag), dtype="parent")

                endpoint = ET.SubElement(tag, "endpoints", name="endpoints", dtype="list" )
                endpoint.text = "[%.4f, %.4f]" % (start, stop)

                probability = ET.SubElement(tag, "probability", name="probability", dtype="list" )
                probability.text ="[%.4f]" % (float(event_probability))
     
        # # convert the tree to a string
        # #
        xmlstr = ET.tostring(root, encoding=nft.DEF_CHAR_ENCODING)

        # # convert the string to a pretty print
        # #
        reparsed = minidom.parseString(
            xmlstr).toprettyxml(indent=nft.DELIM_SPACE)

        # open the output file to write
        #
        with open(ofile, nft.MODE_WRITE_TEXT) as writter:

            # write the xml file
            #
            writter.write(reparsed)

        # exit gracefully
        #
        return True

    # method: Xml::add
    #
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur, sym, level, sublevel):

        # check that this is a valid add, by checking if
        # the sym specified is listed in the class mapping
        #
        class_mapping = self.get_valid_sub(self.schema_fname_d)

        # initialize variables to hold child and parent values
        #
        child = None
        parent = None
        Status = False

        # traverse dictionary holding possible sym parent keys and
        # child values
        #
        for key in class_mapping:

            # if the sym specified is found to be a key
            #
            if (sym == key):

                # store parent value
                #
                parent = key
                Status = True
                break

            # get list of children for each parent sym
            #
            value_list = class_mapping.get(key)

            # iterate through list of children
            #
            for value in value_list:

                # if sym is found in value list
                #
                if (sym == value):

                    # store child value
                    #
                    status = True
                    child = value
                    parent = key
                    break

        # if label is not valid
        #
        if status is False:
            print("Error: %s (line: %s) %s: invalid label (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, sym))

        # if the sym to add is a child we make sure that the parent
        # label already exists in the duration provided
        #
        if (child != None):

            graph = self.get_graph()

            # obtain parent dictionary which is one sublevel below child
            #
            chan_dict = graph[level][sublevel-1]

            # check if parent sym exists at duration
            #
            add_parent = True

            for key in chan_dict:

                # list of events
                #
                value_list = chan_dict.get(key)

                # for each event
                # example of event: [0.0, 10.2787, {'bckg': 1.0}]
                #
                for event in value_list:

                    # get label of event for example ['bckg'] or ['seiz']
                    #
                    event_key = list(event[2].keys())

                    # if the parent event exists at duration
                    #
                    if ((event[1] == dur) and (parent == event_key[0])):

                        Add_parent = False

            # if parent label was not found at duration
            #
            if (add_parent == True):

                # add parent label
                #
                self.graph_d.add(dur, parent, level, sublevel - 1)

        return self.graph_d.add(dur, sym, level, sublevel)

    #
    # end of method

    # method: Xml::delete
    #
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes events of type sym.
    #
    def delete(self, sym, level, sublevel):
        return self.graph_d.delete(sym, level, sublevel)
    #
    # end of method

    # method: Xml::get_graph (UNTESTED)
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()

    # method: Xml::set_graph (UNTESTED)
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the class data to graph
    #
    def set_graph(self, graph):
        return self.graph_d.set_graph(graph)
    #
    # end of method

    # method: Xml::tree_to_dict
    #
    # arguments:
    #  root (xml.etree.ElementTree.root): root of the xml file
    #  uniquekey (string): this key will be used to call each tag in xml file
    #  proplist (list): properties which will be used as attributes
    #  value (string): what the values of each tag should be called,
    #                  it is similar to text in xml files
    #
    # return:
    #  treedict: dictionary equivalent of xml tree
    #
    def tree_to_dict(self, root):

        # check if root is empty with no tag
        #
        if not hasattr(root, TMPL_ROOT):
            print("Error: %s (line: %s) %s: file with no tag" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            return None

        # define a dictionary to be returned
        #
        treedict = defaultdict(list)
        
        # access all the channel
        #
        for montage_channel in root.findall("label/montage_channels/channel"):

            for num, channel in self.chan_map_d.items():
                if channel == montage_channel.attrib["name"]:
                    channel_num = num

            # all of it are string
            #
            for event in montage_channel.findall("event"):
                tag = event.attrib["name"]
                probability = event.find("probability").text.strip("[").strip("]")
                start_time, end_time = \
                    event.find("endpoints").text.strip("[").strip("]").split(", ")

                treedict[channel_num].append([float(start_time), float(end_time), {tag: float(probability)}])

        return treedict

    # method: Xml::validate
    #
    # arguments:
    #  fname: filename to be validated
    #
    # return: a boolean value indicating status
    #
    # This method validates xml file with a schema
    #
    def validate(self, fname):

        # load the schema only once
        #
        if self.schema_d is None:

            # turn a file to XML Schema validator
            #
            self.schema_d = etree.XMLSchema(file=self.schema_fname_d)
            if self.schema_d == None:
                print("Error: %s (line: %s) %s: error loading schema (%s)" %
                      (__FILE__, ndt.__LINE__, ndt.__NAME__,
                       self.schema_fname_d))
                return False

        # parse an XML file
        #
        try:
            xml_file = etree.parse(fname)

        # check for a syntax error
        #
        except etree.XMLSyntaxError as e:
            print("Error: %s (line: %s) %s: xml syntax error (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False

        # check if there was an OSerror (e.g,, file doesn't exist)
        #
        except OSError:
            print("Error: %s (line: %s) %s: xml file doesn't exist (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
            return False

        # validate the schema
        #
        status = self.schema_d.validate(xml_file)
        if status == False:
            try:
                self.schema_d.assertValid(xml_file)
            except etree.DocumentInvalid as errors:
                print("Error: %s (line: %s) %s: %s (%s)" %
                      (__FILE__, ndt.__LINE__, ndt.__NAME__, errors, fname))

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: Xml::add_event
    #
    # arguments: parent_node - the direct parent in the tree
    #            parent - the parent label
    #            child - the child label of the parent
    #            event_mapping - dict mapping of label and events
    #
    # return: new_node - new tree node added to parent node
    #
    # This method adds a new subtag to the parent tag by extracting
    # the events from the parent->child of the event mapping
    #
    def add_event(self, parent_node, parent, child, event_mapping):

        # add a new parent node
        #
        new_node = etree.SubElement(parent_node, child)

        # set the proper attributes
        #
        new_node.set(DEF_NAME, child)
        new_node.set(DEF_TYPE, PARENT_TYPE)

        # add the endpoints
        #
        endpoints = etree.SubElement(new_node, DEF_ENDPTS)

        # set the proper attributes
        #
        endpoints.set(DEF_NAME, DEF_ENDPTS)
        endpoints.set(DEF_TYPE, LIST_TYPE)

        # set the value
        #
        endpoints.text = str(event_mapping[parent][child][0])

        # add the probability
        #
        probs = etree.SubElement(new_node, DEF_PROB)

        # set the proper attributes
        #
        probs.set(DEF_NAME, DEF_PROB)
        probs.set(DEF_TYPE, LIST_TYPE)

        # set the value
        #
        probs.text = str(event_mapping[parent][child][1])

        # return the new node
        #
        return new_node
    #
    # end of function

    # method: Xml::update_montage
    #
    # argument:
    #  montage_file: montage file
    #
    # return: a boolean value indicating status
    #
    # This method updates the chan_map_d and montage_fname_b based on
    # the input montage
    #
    def update_montage(self, montage_file):

        # update new montage file, if input montage file is None, update
        # with the default montage
        #
        if montage_file is None or montage_file == "None":
            self.montage_fname_d = nft.get_fullpath(DEFAULT_MONTAGE_FNAME)
        else:
            self.montage_fname_d = nft.get_fullpath(montage_file)

        montage_fp = open(nft.get_fullpath(self.montage_fname_d),
                          nft.MODE_READ_TEXT)
        try:
            self.chan_map_d = {int(-1): "TERM"}
            for line_mont in montage_fp:
                if line_mont.startswith(DELIM_LBL_MONTAGE):
                    chan_num, name, montage_line = \
                        self.parse_montage(line_mont)
                    self.chan_map_d[chan_num] = name
        except:
            print("Error: %s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, Xml.__CLASS_NAME__,
                   ndt.__NAME__, "error parsing montage file",
                   line_mont))
            montage_fp.close()
            return False
        return True
    #
    # end of method

    # method: Xml::parse_montage
    #
    # arguments:
    #  line: line from label file containing a montage channel definition
    #
    # return:
    #  channel_number: an integer containing the channel map number
    #  channel_name: the channel name corresponding to channel_number
    #  montage_line: entire montage def line read from file
    #
    # This method parses a montage line into it's channel name and number.
    # Splitting a line by two values easily allows us to get an exact
    # value/string from a line of definitions
    #
    def parse_montage(self, line):

        # display an informational message
        #
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s: parsing montage by channel name, number" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # split between '=' and ',' to get channel number
        #
        channel_number = int(
            line.split(nft.DELIM_EQUAL)[1].split(nft.DELIM_COMMA)[0].strip())

        # split between ',' and ':' to get channel name
        #
        channel_name = line.split(
            nft.DELIM_COMMA)[1].split(nft.DELIM_COLON)[0].strip()

        # remove chars from montage line
        #
        montage_line = line.strip().strip(nft.DELIM_NEWLINE)

        # exit gracefully
        #
        return [channel_number, channel_name, montage_line]
    #
    # end of method

    # method: Xml::parse_events
    #
    # arguments:
    #  xml_map_fname: filename of xml map
    #
    # return:
    #  a boolean indicating status
    #
    # This method parses a map file to get xml events which
    # contains parent labels
    #
    def parse_events(self, xml_map_fname=None):

        if xml_map_fname is not None:
            self.xml_map_fname_d = xml_map_fname

        # open file with read
        #
        fp = open(self.xml_map_fname_d, nft.MODE_READ_TEXT)

        lines = fp.readlines()

        self.xml_events = {}

        for line in lines:
            if line.startswith(DEF_XML_EVENTS):
                # remove all characters to remove, and split by ','
                #
                events = ''.join(c for c in line.split(nft.DELIM_EQUAL)[1]
                                 if c not in REM_CHARS)

                events = events.split(nft.DELIM_COMMA)

                # create a dict from string, split by ':'
                #   e.g. 'null': 'null_type' -> mappings[null] = 'null_type'
                #
                for s in events:
                    self.xml_events[str(s.split(nft.DELIM_COLON)[0])] \
                        = s.split(nft.DELIM_COLON)[1]

        return True

# class: Ann
#
# This class is the main class of this file. It contains methods to
# manipulate the set of supported annotation file formats including
# label (.lbl) and time-synchronous events (.tse) formats.
#
class AnnEeg:

    # method: AnnEeg::constructor
    #
    # arguments: none
    #
    # return: none
    #
    # This method constructs AnnEeg
    #
    def __init__(self):

        # set the class name
        #
        AnnEeg.__CLASS_NAME__ = self.__class__.__name__

        # declare variables for each type of file:
        #  these variable names must match the FTYPES declaration.
        #
        self.tse_d = Tse()
        self.lbl_d = Lbl()
        self.csv_d = Csv(None)
        self.xml_d = Xml(None, None)

        # declare variable to store type of annotations
        #
        self.type_d = None
    #
    # end of method

    # method: AnnEeg::load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a boolean value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname, schema=None, montage_f=None, map_f=None):

        # reinstantiate objects, this removes the previous loaded annotations
        #
        self.lbl_d = Lbl()
        self.tse_d = Tse()
        self.csv_d = Csv(montage_f)
        self.xml_d = Xml(schema, map_f)

        # determine the file type
        #
        magic_str = nft.get_version(fname)

        self.type_d = self.check_version(magic_str)

        if self.type_d == None or self.type_d == False:
            print("Error: %s (line: %s) %s: unknown file type (%s: %s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname, magic_str))
            return False

        # load the specific type
        #
        return getattr(self, FTYPES[self.type_d][1]).load(fname)

    #
    # end of method

    # method: AnnEeg::get
    #
    # arguments:
    #  level: the level value
    #  sublevel: the sublevel value
    #
    # return:
    #  events: a list of ntuples containing the start time, stop time,
    #          a label and a probability.
    #
    # This method returns a flat data structure containing a list of events.
    #
    def get(self, level=int(0), sublevel=int(0), channel=int(-1)):

        if self.type_d is not None:
            events = getattr(self,
                             FTYPES[self.type_d][1]).get(level, sublevel,
                                                         channel)
        else:
            print("Error: %s (line: %s) %s: no annotation loaded" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            return False

        # exit gracefully
        #
        return events
    #
    # end of method

    # method: AnnEeg::display
    #
    # arguments:
    #  level: level value
    #  sublevel: sublevel value
    #  fp: a file pointer (default = stdout)
    #
    # return: a boolean value indicating status
    #
    # This method displays the events at level/sublevel.
    #
    def display(self, level=int(0), sublevel=int(0), fp=sys.stdout):

        if self.type_d is not None:

            # display events at level/sublevel
            #
            status = getattr(self,
                             FTYPES[self.type_d][1]).display(level,
                                                             sublevel, fp)

        else:
            sys.stdout.write("Error: %s (line: %s) %s %s" %
                             (ndt.__NAME__, ndt.__LINE__, ndt.__NAME__,
                              "no annotations to display"))
            return False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of annotation to write
    #  sublevel: sublevel of annotation to write
    #
    # return: a boolean value indicating status
    #
    # This method writes annotations to a specified file.
    #
    def write(self, ofile, level=int(0), sublevel=int(0)):

        # write events at level/sublevel
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[self.type_d][1]).write(ofile,
                                                                 level,
                                                                 sublevel)
        else:
            sys.stdout.write("Error: %s (line: %s) %s: %s" %
                             (__FILE__, ndt.__LINE__, ndt.__NAME__,
                              "no annotations to write"))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::add
    #
    # arguments:
    #  dur: duration of file
    #  sym: symbol of event to be added
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method adds events to the current events based on args.
    #
    def add(self, dur, sym, level, sublevel):

        # add labels to events at level/sublevel
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[self.type_d][1]).add(dur,
                                                               sym,
                                                               level,
                                                               sublevel,)
        else:
            print("Error: %s (line: %s) %s: no annotations to add to" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::delete
    #
    # arguments:
    #  sym: symbol of event to be deleted
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a boolean value indicating status
    #
    # This method deletes all events of type sym
    #
    def delete(self, sym, level, sublevel):

        # delete labels from events at level/sublevel
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[self.type_d][1]).delete(sym,
                                                                  level,
                                                                  sublevel)
        else:
            print("Error: %s (line: %s) %s: no annotations to delete" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::set_type
    #
    # arguments:
    #  type: the type of ann object to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the type and graph in type from self.type_d
    #
    def set_type(self, ann_type):

        # set the graph of ann_type to the graph of self.type_d
        #
        if self.type_d is not None:
            status = getattr(self,
                             FTYPES[ann_type][1]).set_graph(
                                 getattr(self,
                                         FTYPES[self.type_d][1]).get_graph())
            self.type_d = ann_type

        else:
            print("Error: %s (line: %s) %s: no graph to set" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: AnnEeg::set_graph
    #
    # arguments:
    #  type: type of ann object to set
    #
    # return: a boolean value indicating status
    #
    # This method sets the type and graph in type from self.type_d
    #
    def set_graph(self, graph):
        status = getattr(self, FTYPES[self.type_d][1]).set_graph(graph)
        return status

    # method: Anneeg:: delete_graph
    #
    #
    def delete_graph(self):
        getattr(self, FTYPES[self.type_d][1]).graph_d.delete_graph()
        return True

    # method: AnnEeg::get_graph
    #
    # arguments: none
    #
    # return: the entire annotation graph
    #
    # This method returns the entire stored annotation graph
    #
    def get_graph(self):

        # if the graph is valid, get it
        #
        if self.type_d is not None:
            graph = getattr(self, FTYPES[self.type_d][1]).get_graph()
        else:
            print("Error: %s (line: %s) %s: no graph to get" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
            graph = None

        # exit gracefully
        #
        return graph
    #
    # end of method

    # method: AnnEeg::check_version
    #
    # arguments:
    #  magic: a magic sequence
    #
    # return: a character string containing the name of the type
    #
    def check_version(self, magic):

        # check for a match
        #
        for key in FTYPES:
            if FTYPES[key][0] == magic:
                return key

        # exit (un)gracefully:
        #  if we get this far, there was no match
        #
        return False
    #
    # end of method


    # method:: AnnEeg::is_csv
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
                  (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                   ndt.__NAME__, fname))

        # open the file
        #
        fp = open(fname, nft.MODE_READ_TEXT)
        if fp is None:
            print("Error: %s (line: %s) %s::%s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                   ndt.__NAME__, fname))
            return False

        # read the first line in the file
        #
        header = fp.readline()
        if dbgl > ndt.BRIEF:
            print("%s (line: %s) %s::%s: header (%s)" %
                  (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                   ndt.__NAME__, header))
        fp.close()

        # exit gracefully:
        #  if the beginning of the file is the magic sequence
        #  then it is an imagescope xml file
        if DEF_CSV_HEADER in header.strip():
            return True
        else:
            if dbgl > ndt.BRIEF:
                print("Error: %s (line: %s) %s::%s: processing error (%s)" %
                      (__FILE__, ndt.__LINE__, AnnEeg.__CLASS_NAME__,
                       ndt.__NAME__, fname))
            return False
#
# end of class

#
# end of file
