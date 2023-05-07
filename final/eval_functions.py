import pandas as pd


data = pd.read_csv("/Users/gavinkoma/Desktop/pattern_rec/final/data_s14/dev/13/f013000.csv")
print(data.head())
train = pd.read_csv("/Users/gavinkoma/Desktop/pattern_rec/final/data_s14/train/02/f002000.csv")
print(train.head())


from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix,\
                            classification_report, \
                            log_loss, \
                            accuracy_score
from sklearn import metrics
import csv

model1 = MLPClassifier(hidden_layer_sizes=(30,15,10,5),
             	     activation="tanh",
                	 random_state=1,
               	  	 max_iter=2000)
	
x_train = train.drop(['class'], axis=1)
y_train = train['class']
#print(type(model).__name__)
model.partial_fit(x_train,y_train,classes=[0,1,2,3,4])







