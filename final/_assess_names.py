##### Gavin Koma #####
import os
import glob



# print('Loop over dirs and files:')
train_path = '/Users/gavinkoma/Desktop/pattern_rec/final/subset_data'
dev_path = '/Users/gavinkoma/Desktop/pattern_rec/final/'
training_sets = []
dev_sets = []

file = glob.glob(train_path + "/*/*")
for _file in file:
    training_sets.append(_file)

print(training_sets)

# for root, dirs, files in os.walk(train_path):
#     #print(root)
#     for _file in files:
#         # print(str(root)+str(dirs).replace("[]","/")+str(_file))
#         if '.csv' in _file:
#             training_sets.append(str(root) + str(dirs).replace("[]", "/") + str(_file))
#             #print(training_sets)
#
#
# for root, dirs, files in os.walk(dev_path):
#     #print(root)
#     for _file in files:
#         if '.csv' in _file:
#             # print(str(root)+str(_file))
#             dev_sets.append(str(root) + str(dirs).replace("[]", "/") + str(_file))
