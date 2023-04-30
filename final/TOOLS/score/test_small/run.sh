#/bin/sh

# set this variable to the location of your install
#
export NEDC_NFC=/data/isip/www/isip/courses/temple/ece_8527/resources/data/set_14/TOOLS/score;
export PYTHONPATH="$NEDC_NFC/lib:."

# score the data
#
../src/nedc_eval_eeg.py ref.list hyp.list

# look at the output
#
cat output/summary.txt

#
# end of file

