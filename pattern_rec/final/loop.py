import glob

print('Loop over dirs and files:')
train_path = '/Users/gavinkoma/Desktop/pattern_rec/final/train_dev/'
dev_path = '/Users/gavinkoma/Desktop/pattern_rec/final/data_s14/train'
training_sets = []
dev_sets = []

file_train = glob.glob(train_path + "/*/*")
for _file in file_train:
    #print(_file)
    if '.csv' in _file:
        training_sets.append(_file)

print(training_sets)