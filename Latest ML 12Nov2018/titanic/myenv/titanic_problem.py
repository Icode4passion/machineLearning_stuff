# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:30:17 2018

@author: usunkesu
"""

'''
Tutorial reference from 
Topic : Turning Machine Learning Models into APIs in Python
url :  https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

'''
import pandas as pd
#import numpy as np
url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)
include = ['Age','Sex','Embarked','Survived']
df_ = df[include]


catageoricals = []
for col , col_type in df_.dtypes.iteritems():
    if col_type == 'O':
        catageoricals.append(col)
    else:
        df_[col].fillna(0,inplace=True)


#s = pd.Series(list('abc'))
#print(s)
#s = pd.get_dummies(s)
#print(s)

#One hot encoding 
        
df_ohe = pd.get_dummies(df_,columns=catageoricals,dummy_na=True)

#Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

#save your model Serialization and Deserialization

from sklearn.externals import joblib
joblib.dump(lr,'model.pkl')
print('Model dumped')

#load the model that you save

lr = joblib.load('model.pkl')

# Saving the data columns from training

model_columns = list(x.columns)
joblib.dump(model_columns,'model_columns.pkl')
print("Models columns dumped!")


















