import os
import glob
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, classification_report, accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt

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
         # print(chunks[len(chunks) -1])

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
     #print(df.head())
     return df

def build_main_df(filename):
     return pd.read_csv(filename, comment='#', header=None, names=['t', 'signal'])

def load_data():
    #print('Loop over dirs and files:')
    train_path = '/Users/gavinkoma/Desktop/pattern_rec/final/data_s14/train'
    dev_path = '/Users/gavinkoma/Desktop/pattern_rec/final/data_s14/dev'
    training_sets = []
    dev_sets = []

    file_train = glob.glob(train_path + "/*/*")
    for _file in file_train:
        print(_file)
        if '.csv' in _file:
            training_sets.append(_file)


    file_dev = glob.glob(dev_path + "/*/*")
    for _file in file_dev:
        print(_file)
        if '.csv' in _file:
            dev_sets.append(_file)
    
    # for root, dirs, files in os.walk(train_path):
    #     print(root)
    #     for _file in files:
    #         #print(str(root)+str(dirs).replace("[]","/")+str(_file))
    #         if '.csv' in _file:
    #             training_sets.append(str(root)+str(dirs).replace("[]","/")+str(_file))
                                                          
    # for root, dirs, files in os.walk(dev_path):
    #     print(root)
    #     for _file in files:
    #         if '.csv' in _file:
    #         #print(str(root)+str(_file))
    #             dev_sets.append(str(root)+str(dirs).replace("[]","/")+str(_file))

    return training_sets, dev_sets
        


def process_data(filename, training=True):
     main_df = build_main_df(filename)
     if training:
          main_df = add_meta_data(filename, main_df)
     main_df = add_feature_engineering(main_df)
     return main_df
 
def prepare_load():
    training_files, testing_files = load_data()
    df_arr_train = []
    i=0
    for elem in training_files:
        df_arr_train.append(process_data(elem))
        print('training ', i, 'file:', i, 'of', len(training_files))
        i += 1

    df_arr_test = []
    i = 0
    for elem in testing_files:
        df_arr_test.append(process_data(elem))
        print('testing ', i, 'file:', i, 'of', len(testing_files))
        i += 1

    #df_arr_train = [process_data(x) for x in training_files]
    #df_arr_test = [process_data(x) for x in testing_files]

    return df_arr_train, df_arr_test

def train(df_arr,model,partial_fit):
    if partial_fit:
        i=0
        for data in df_arr:
            x_train = data.drop(['class'], axis=1)
            y_train = data['class']
            #print(type(model).__name__)
            model.partial_fit(x_train,y_train,classes=[0,1,2,3,4])
        return model

    else:
        data = pd.concat(df_arr)
        x_train = data.drop(['class'], axis=1)
        y_train = data['class']
        #print(data.size)
        model.fit(x_train,y_train)
        return model

def predictions(df_arr,model):
    data = pd.concat(df_arr)
    x_eval = data.drop(['class'], axis=1)
    y_eval = data['class']
    y_pred = model.predict(x_eval)
    #print()
    #print(pd.DataFrame(y_pred))
    confusion_matrix = multilabel_confusion_matrix(y_eval,y_pred)
    print("multi cm \n", confusion_matrix)
    #precision is true positives / sum(true positive + false positive)
    #what percent of guess positives are correct

    #recall is tru positives/ sum(true pos + false neg)
    #percent of guess positives that were not missed

    #f1 is harmonic mean of the above two values
    print("class report \n8 ", classification_report(y_eval,y_pred))
    print("accuracy: ", accuracy_score(y_eval,y_pred))
    print("error: ", 1-accuracy_score(y_eval,y_pred))

    return



def main():
    model1 = MLPClassifier(hidden_layer_sizes=(30,15,10,5),
                          activation="relu",
                          random_state=1,
                          max_iter=2000)
    model2 = RandomForestClassifier(max_depth=3,
                                   random_state=0)
    df_arr_train, df_arr_test = prepare_load()
    print('Data loaded!')

    #Train Models
    df_arr = df_arr_train
    print('Start MLP Training...')
    model1 = train(df_arr, model1, True)
    #print('MLP Trained, Starting Random Forest Training')
    #model2 = train(df_arr, model2, False)
    print('Both models trained, scoring starting...')

    #Test Models
    df_arr = df_arr_test

    #Predictions
    predictions(df_arr, model1)
    #predictions(df_arr, model2)

    return

main()

# if __name__ == '__main__':
#      import sys
#      try:
#           process_data(sys.argv[1])
#      except Exception as e:
#           print('failure encountered', e)







