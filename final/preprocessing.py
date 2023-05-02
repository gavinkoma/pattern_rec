import pandas as pd


def add_meta_data(filename, df):
     with open(filename, 'r') as reader:
          file = reader.readlines()
          chunks = [x for x in file if '#' in x]
          chunks = chunks[3:len(chunks) - 1]
          chunks = [
               {
                    'chunk': int(x.split('# ')[1].split(':')[0]),
                    'class': int(x.split('class: ')[1].split(',')[0]),
                    'start': int(x.split('start: ')[1].split(',')[0]),
                    'end': int(x.split('stop = ')[1].split('\n')[0])
               } for x in chunks
          ]
          print(chunks[len(chunks) -1])

     def check_value(val, feature):
          for elem in chunks:
               if val >= elem['start'] and val <= elem['end']:
                    return elem[feature]

     df['class'] = df['t'].apply(lambda x: check_value(x, 'class'))
     df['chunk_num'] = df['t'].apply(lambda x: check_value(x, 'chunk'))
     df['chunk_start'] = df['t'].apply(lambda x: check_value(x, 'start'))
     df['chunk_end'] = df['t'].apply(lambda x: check_value(x, 'end'))

     return df

def add_feature_engineering(df):
     def add_neighbors(col_name, shift, df):
          r = range(0, shift) if shift > 0 else range(shift, 0)
          for i in r:
               df[f'neighbor_{i}'] = df[col_name].shift(i * -1).fillna(0)
          return df

     def add_neighbor_avg(col_name, intervals, df):
          for i in intervals:
               df[f'{i}_neighbor_avg'] = df[col_name].rolling(
                    i, 
                    min_periods=1, 
                    center=True
               ).mean()
          return df

     def add_neighbor_std(col_name, intervals, df):
          for i in intervals:
               df[f'{i}_neighbor_avg'] = df[col_name].rolling(
                    i, 
                    min_periods=1, 
                    center=True
               ).std()
          return df

     intervals = [10,20,50,100,300,600,1200]

     df = add_neighbors('signal', 10, df)
     df = add_neighbors('signal', -10, df)
     df = add_neighbor_avg('signal', intervals, df)
     df = add_neighbor_std('signal', intervals, df)

     return df

def build_main_df(filename):
     return pd.read_csv(filename, comment='#', header=None, names=['t', 'signal'])

def process_data(filename, training=True):
     main_df = build_main_df(filename)
     if training:
          main_df = add_meta_data(filename, main_df)
     main_df = add_feature_engineering(main_df)
     return main_df

def train_model(directory, tuned_model, recursive=False):
     #iterate through directories/files
     #training_sets = [list of filenames to be trained on]
     #testing_sets = set(random.choices(k=len(training) * .30))
     #training_set = set(training_set) - testing_set
     #for file in training_sets:
          #df = process_data(file)
          #X, y = {logic goes here}
          #tuned_model.partial_fit(X_train, y_train, classes=np.unique(y), warm_start=False)
     return tuned_model, testing_sets



if __name__ == '__main__':
     import sys
     try:
          process_data(sys.argv[1])
     except Exception as e:
          print('failure encountered', e)







