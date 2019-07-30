# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:59:07 2018

@author: usunkesu

tutorial based on the url

https://www.datacamp.com/community/tutorials/kaggle-tutorial-machine-learning


"""

"""
Importing Libraries
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer    


sns.set()

"""

Reading Data Train and Test

"""

titanic_train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")

titanic_test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")



######################


# Removing Survived field from train data

servived_data = titanic_train.Survived

# Concatinating Train and test data. (make sure that any operations that
# you perform on the training set are also being done on the test data set)

data = pd.concat([titanic_train.drop(['Survived'],axis=1),titanic_test])

#data.info()


# Filling the missed values using imputing

#impute_features = ['Age','Fare']
#compute_imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
#data[impute_features]=compute_imputer.fit_transform(data[impute_features])
#data.info()


# Or other way of imouting in simple way is 

data['Age']=data.Age.fillna(data.Age.median())
data['Fare']=data.Fare.fillna(data.Fare.median())
data.info()

#encoding data with numbers for Catogeorical data like Sex and Embarked


data = pd.get_dummies(data,columns=['Sex'])
data.head()


['Sex_male','Fare','Age','Pclass','SibSp']



