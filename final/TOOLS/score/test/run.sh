#/bin/sh

# set this variable to the location of your install
#
export NEDC_NFC=/data/isip/www/isip/courses/temple/ece_8527/resources/data/set_14/TOOLS/score;
export PYTHONPATH="$NEDC_NFC/lib:."

# score the training data
#
../src/nedc_eval_eeg.py -o output_train lists/ref_train.list lists/hyp_train.list

# score the dev data
#
../src/nedc_eval_eeg.py -o output_dev lists/ref_dev.list lists/hyp_dev.list

#
# end of file

