import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv('data.csv')
data.head()
y = data.diagnosis                          # M or B 
list = ['Unnamed: 32','id','diagnosis']
x = data.drop(list,axis = 1 )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)

print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)

#To get names of features
filter =select_feature.get_support()
features = x_train.columns
print(features[filter])


x_train_2 = select_feature.transform(x_train)
print(x_train_2)
x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)

 

pickle.dump(clr_rf_2,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))