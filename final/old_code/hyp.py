import pandas as pd
import glob
import os

#this fuckin needs to output our hyp file 

names = []
name_path = '/Users/gavinkoma/Desktop/pattern_rec/final/scored_data'
file_path = glob.glob(name_path+"/*")
for _file in file_path:
    #print(_file)
    if '.csv' in _file:
        names.append(_file)
files_ = []
for file in names:
    file_name = os.path.basename(file)
    #print(file_name)
    files_.append(file_name)
#print(files_)

for file in files_:
	s='scored_'
	file.replace('scored_','')
	print(file)


# for file in files_:
# 	data = pd.read_csv(f"/Users/gavinkoma/Desktop/pattern_rec/final/scored_data/{file}")
# 	gd=data.groupby(['start_time']).apply(lambda gdf: gdf[0:1].reset_index().join( gdf[['label']].mode().add_suffix('_mode') ))

# 	# reset index added this column we dont want
# 	gd2=gd.drop(columns='index')
# 	gd2=gd2.drop(columns='label')

# 	# we needed to drop the new multiindex
# 	gd2=gd2.reset_index(drop=True)
# 	#print(gd2.head())
# 	#cols = list(gd2.columns.values)
# 	#print(cols)

# 	gd2 = gd2.iloc[:,[0, 1, 2, 4, 3]]
# 	gd2 = gd2.rename(columns={'channel': 'channel',
# 								'start_time': 'start_time',
# 								'stop_time' : 'stop_time',
# 								'label_mode' : 'label',
# 								'confidence' : 'confidence'})
# 	#print(gd2.head())

# 	hypothesis = pd.DataFrame(gd2).to_csv(f"/Users/gavinkoma/Desktop/pattern_rec/final/hypothesis_files/{file}",index = False)
