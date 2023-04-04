#!/usr/bin/env python
#
# file: $NEDC_NFC/class/python/nedc_eval_tools/nedc_eval_common.py
#
# revision history:
#
# 20200813 (LV): added parse_files method
# 20200622 (LV): first version
#
# Usage:
#  import nedc_eval_common as nec
#
# This file contains a collection of functions and variables commonly used
# across EEG evaluation tools.
#------------------------------------------------------------------------------

# import system modules
#
import os
from operator import itemgetter
from pathlib import Path
import sys

# import nedc_modules
#
import nedc_ann_eeg_tools as nae
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

# define symbols to use to create filler events:
#  a background symbol
#  a duration key
#  a default probability (as a string)
#  a precision used to compare floats
#
BCKG = "0"
DURATION_KEY = "duration"
PROBABILITY = "1.0000"
PRECISION = int(4)

# define a constant used to indicate a null choice  
#                                                                              
NULL_CLASS = "***"

# define constant that appears in the parameter file                          
#                                                                              
PARAM_MAP = "MAP"

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()

# function: format_hyp
#
# arguments:
#  ref: the references events as a list
#  hyp: the hypothesis events as a list
#
# return:
#  refo: a string displaying the alignment of the reference
#  hypo: a string displaying the alignment of the hypothesis
#  hits: the number of correct
#  subs: the number of substitution errors
#  inss: the number of insertion errors
#  dels: the number of deletion errors
#
# This function displays all the results in output report.
#
def format_hyp(ref, hyp):

    # declare return values
    #
    hits = int(0)
    subs = int(0)
    inss = int(0)
    dels = int(0)
    
    # find the max label length and increment by 1
    #
    maxl = int(0)
    for lbl in ref:
        if len(lbl) > maxl:
            maxl = len(lbl)
    for lbl in hyp:
        if len(lbl) > maxl:
            maxl = len(lbl)
    maxl += 1

    # loop over the input: skip the first and last label
    #
    refo = nft.STRING_EMPTY
    hypo = nft.STRING_EMPTY

    for i in range(1, len(ref)-1):

        # save a copy of the input
        #
        lbl_r = ref[i]
        lbl_h = hyp[i]

        # count the errors
        #
        if (ref[i] == NULL_CLASS) and (hyp[i] != NULL_CLASS):
            inss += int(1)
            lbl_h = hyp[i].upper()
        elif (ref[i] != NULL_CLASS) and (hyp[i] == NULL_CLASS):
            dels += int(1)
            lbl_r = ref[i].upper()
        elif (ref[i] != hyp[i]):
            subs += int(1)
            lbl_r = ref[i].upper()
            lbl_h = hyp[i].upper()
        else:
            hits += int(1)
            
        # append the strings
        #
        refo += ("%*s " % (maxl, lbl_r))
        hypo += ("%*s " % (maxl, lbl_h))

    # exit gracefully
    #
    return (refo, hypo, hits, subs, inss, dels)
#
# end of function

# function: create_table
#
# arguments:
#  cnf: confusion matrix
#
# return: 
#  header: header for the table
#  tbl: a table formatted for print_table
#
# This function transforms a confusion matrix into a format
# required for print_table.
#
def create_table(cnf):

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: formatting confusion matrix for printing" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # declare local variables
    #
    tbl = []

    # create the header output by loop over the second index of the input
    #
    header = ['Ref/Hyp:']
    for key1 in cnf:
        for key2 in cnf[key1]:
            header.append(key2)
        break

    # loop over each key and then each row
    #
    counter = int(0)
    for key1 in cnf:

        # append the header
        #
        tbl.append([key1])

        # compute the sum of the entries in the row
        #
        sum = float(0)
        for key2 in cnf[key1]:
            sum += float(cnf[key1][key2])
        
        # transfer counts and percentages to the output table:
        #  note there is a chance the counts are zero due to a bad map
        #
        for key2 in cnf[key1]:
            if sum == 0:
                val1 = float(0.0)
                val2 = float(0.0)
            else:
                val1 = float(cnf[key1][key2])
                val2 = float(cnf[key1][key2]) / sum * 100.0
            tbl[counter].append([val1, val2])

        # increment the counter
        #
        counter += 1
            
    # exit gracefully
    #
    return header, tbl
#
# end of function

# function: print_table
#
# arguments: 
#  title: the title of the table
#  headers: the column headers
#  data: a list containing the row-by-row entries
#  fmt_lab: the format specification for a label (e..g, "%10s")
#  fmt_cnt: the format specification for the 1st value (e.g., "%8.2f")
#  fmt_pct: the format specification for the 2nd value (e.g., "%6.2f")
#  fp: the file pointer to write to
#
# return: a boolean value indicating the status
#
# This function prints a table formatted in a relatively standard way.
# For example:
#
#     title = "This is the title"
#     headers = ["Ref/Hyp:", "Correct", "Incorrect"]
#     data = ["seiz:", [ 8.00, 53.33],  [7.00, 46.67]],\
#            ["bckg:", [18.00, 75.00],  [6.00, 25.00]],\
#            ["pled:", [ 0.00,  0.00],  [0.00,  0.00]]]
#
# results in this output:
#
#              This is the title             
#  Ref/Hyp:     Correct          Incorrect     
#     seiz:    8.00 ( 53.33%)    7.00 ( 46.67%)
#     bckg:   18.00 ( 75.00%)    6.00 ( 25.00%)
#     pled:    0.00 (  0.00%)    0.00 (  0.00%)
#
# fmt_lab is the format used for the row and column headings. fmt_cnt is
# used for the first number in each cell, which is usually an unnormalized
# number such as a count. fmt_pct is the format for the percentage.
#
def print_table(title, headers, data,
                fmt_lab, fmt_cnt, fmt_pct, fp):

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: printing table" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # get the number of rows and colums for the numeric data:
    #  the data structure contains two header rows
    #  the data structure contains one column for headers
    #
    nrows = len(data);
    ncols = len(headers) - 1

    # get the width of each colum and compute the total width:
    #  the width of the percentage column includes "()" and two spaces
    #
    width_lab = int(float(fmt_lab[1:-1]))
    width_cell = int(float(fmt_cnt[1:-1]))
    width_pct = int(float(fmt_pct[1:-1]))
    width_paren = 4
    total_width_cell = width_cell + width_pct + width_paren
    total_width_table = width_lab + \
                        ncols * (width_cell + width_pct + width_paren);

    # print the title
    #
    fp.write("%s".center(total_width_table - len(title)) % title)
    fp.write(nft.DELIM_NEWLINE)

    # print the first heading label right-aligned
    #
    fp.write("%*s" % (width_lab, headers[0]))

    # print the next ncols labels center-aligned:
    #  add a newline at the end
    #
    for i in range(1, ncols + 1):

        # compute the number of spaces needed to center-align
        #
        num_spaces = total_width_cell - len(headers[i])
        num_spaces_2 = int(num_spaces / 2)

        # write spaces, header, spaces
        #
        fp.write("%s" % nft.DELIM_SPACE * num_spaces_2)
        fp.write("%s" % headers[i])
        fp.write("%s" % nft.DELIM_SPACE * (num_spaces - num_spaces_2))
    fp.write(nft.DELIM_NEWLINE)

    # write the rows with numeric data:
    #  note that "%%" is needed to print a percent
    #
    fmt_str = fmt_cnt + " (" + fmt_pct + "%%)"

    for d in data:

        # write the row label
        #
        fp.write("%*s" % (width_lab, d[0] + nft.DELIM_COLON))

        # write the numeric data and then add a new line
        #
        for j in range(1,ncols + 1):
            fp.write(fmt_str % (d[j][0], d[j][1]))
        fp.write(nft.DELIM_NEWLINE)

    # exit gracefully
    #
    return True
#
# end of function

# function: parse_files
#
# arguments:
#  reflist: list of hypothesis or reference files
#  scmap: a scoring map used to augment the annotations
#
# return:
#  odict: dictionary with unique filename sequence as key and list of
#  corresponding annotations as the values 
#
# This function parses each file in a list of reference files into a dictionary
#  format with file names as keys. The dictionary is of the format:
#   '0000258_s002_t002': [[0.0, 24.0, {'bckg': 1}], [24.0, 151.0, {'seiz': 1}]
#
def parse_files(files, scmap = None):

    # display informational message                                           
    #                                                                     
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: parsing files" % 
              (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
    # declare local variables
    #
    ann = nae.AnnEeg()
    odict = {}
    
    # load annotations
    #
    for i in range(len(files)):
        
        if ann.load(files[i]) == False:
            print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   ndt.__NAME__, "error loading references",
                   files[i]))
            return False

        # get the events and sort them based on start time
        #
        events = ann.get()
        events_sorted = sorted(events, key=itemgetter(0))
        
        # augment the annotation with background intervals
        #
        events_new = augment_annotation(events_sorted, BCKG)
        
        # store full file path to parse for sequence unique to each file
        #
        fname = Path(files[i]).stem

        # create dictionary with filename as key
        #
        if fname not in odict:
            odict[fname] = events_new
        else:
            # add event to the list
            #
            odict[fname].append(events_new)

    # return a dictionary
    #
    return odict
    
#
# end of function

# function: augment_annotation
#
# arguments:
#  events: an event list
#  sym: the symbol to use for the annotation that fills in the gaps
#
# return: a new event list with the augmented annotation
#
# This method fills in gaps in an annotation with a user-supplied symbol.
#
def augment_annotation(events, sym = BCKG):

    # display informational message                                           
    #                                                                     
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: events (before)" % 
              (__FILE__, ndt.__LINE__, ndt.__NAME__))
        for ev in events:
            print(ev)

    # loop over the events
    #
    events_new = []
    curr_time = float(0.0)

    for ev in events:

        # if the current time equals the start time of the next event,
        # copy the event and advance time to the end of that event
        #
        start_time  = round(ev[0], PRECISION)
        end_time  = round(ev[1], PRECISION)
        key_fdur = list(ev[2].keys())[1]
        fdur = round(float((ev[2])[key_fdur]), PRECISION)

        if curr_time != start_time:

            # add a filler event
            #
            events_new.append([curr_time, start_time,
                               {sym: PROBABILITY, DURATION_KEY: str(fdur)}])
        # append the event
        #
        events_new.append(ev)
        curr_time = end_time

    # add an end of file background event if necessary
    #
    if curr_time != fdur:
        events_new.append([curr_time, fdur,
                           {sym: PROBABILITY, DURATION_KEY: str(fdur)}])

    # display informational message                                           
    #                                                                     
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: events (after)" % 
              (__FILE__, ndt.__LINE__, ndt.__NAME__))
        for ev in events_new:
            print(ev)

    # exit gracefully
    #
    return events_new

    #
    # end of function

# end of file
#
