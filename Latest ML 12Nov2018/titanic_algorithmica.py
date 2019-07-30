# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 20:59:24 2018

@author: usunkesu
"""

import pandas as pd
import numpy as np
from sklearn import model_selection, ensemble ,tree
from sklearn.preprocessing import Imputer , LabelEncoder , OneHotEncoder
import seaborn as sns
from sklearn import feature_selection
#creation of data frames from csv
titanic_train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
print(titanic_train.info())

#read test data
titanic_test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
print(titanic_test.info())

survived_data = titanic_train.Survived

titanic_data = pd.concat((titanic_train.drop(['Survived'],axis=1),titanic_test))
 
titanic_data.info()


#
#titanic.head()

# Filling the NaN Values with mean with out imputer
#titanic.isnull().head()
#(print(titanic.isnull().sum()))
#
#titanic = titanic.fillna(titanic.mean(),inplace = True)
#print(titanic.isnull().sum())


#filling up the missing values for Age and Fare colums using Impute 
impute_features = ['Age','Fare']
compute_imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
compute_imputer=compute_imputer.fit(titanic_data[impute_features])
compute_imputer.statistics_
titanic_data[impute_features]= compute_imputer.fit_transform(titanic_data[impute_features])
print(titanic_train.isnull().sum())
titanic_data['Embarked'] = titanic_data['Embarked'].fillna('S')
print(titanic_train.isnull().sum())

titanic_data.info()


#Encoding or converting Catogerical values (Male or Female ) to Integer vlaue 


label_encoder_sex = LabelEncoder()
titanic_data['Sex']=label_encoder_sex.fit_transform(titanic_data['Sex'])


label_encoder_sex = LabelEncoder()
titanic_data['Embarked']=label_encoder_sex.fit_transform(titanic_data['Sex'])


label_encoder_sex = LabelEncoder()
titanic_data['Pclass']=label_encoder_sex.fit_transform(titanic_data['Sex'])



#Embarked had missing values that ahs been taken care by 
#astype and cat.codes any one is ok
#titanic_train['Embarked']  = titanic_train['Embarked'].astype('category')
#print(titanic_train)
#print(titanic_train.isnull().sum())
#titanic_train['Embarked']  = titanic_train['Embarked'].cat.codes
#print(titanic_train['Embarked'])
#print(titanic_train.isnull().sum())


#ohe_features = ['Age','Sex','Embarked']
#one_hot_encoder = OneHotEncoder()
#titanic_train= one_hot_encoder.fit_transform(titanic_train[ohe_features]).toarray()

# Dividing the data in to test and train data




def plot_feature_importances(estimator, X_train, y_train):
    indices = np.argsort(estimator.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],x = estimator.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("RF feature importances") 


#feat_names = titanic_data.columns.values
#print(feat_names)


rf_selector = ensemble.RandomForestRegressor(random_state=100)
rf_selector.fit(data_train,data_test)
plot_feature_importances(rf_selector, data_train, data_test)
select = feature_selection.SelectFromModel(rf_selector, prefit=True)
select.transform(data_train)


#X_train = pd.get_dummies(X_train,columns=['Sex','Age','Fare'],dummy_na=False)
#y_train = titanic_train['Survived']




dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'max_depth' : [3,4,5,6,7] , 'criterion':['entropy','gini']}
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator,dt_grid,scoring='accuracy',cv=10, refit=True)
dt_grid_estimator.fit(X_train,y_train)

print(dt_grid_estimator.cv_results_)
print(dt_grid_estimator.best_estimator_)
print(dt_grid_estimator.best_score_)
print(dt_grid_estimator.best_params_)

best_dt_estimator = dt_grid_estimator.best_estimator_


# Test data same as Train Data

compute_imputer=compute_imputer.fit(titanic_test[impute_features])
titanic_test[impute_features]= compute_imputer.transform(titanic_test[impute_features])


label_encoder_sex = LabelEncoder()
titanic_test['Sex']=label_encoder_sex.fit_transform(titanic_test['Sex'])

label_encoder_Pclass = LabelEncoder()
titanic_test['Age']=label_encoder_Pclass.fit_transform(titanic_test['Age'])

label_encoder_sex = LabelEncoder()
titanic_train['Fare']=label_encoder_sex.fit_transform(titanic_train['Fare'])

#label_encoder_sex = LabelEncoder()
#titanic_train['Pclass']=label_encoder_sex.fit_transform(titanic_train['Pclass'])

#Embarked had missing values that ahs been taken care by 
#astype and cat.codes any one is ok
#titanic_test['Embarked']  = titanic_test['Embarked'].astype('category')
#titanic_test['Embarked']  = titanic_test['Embarked'].cat.codes

X_test = titanic_test[features]
print(X_test)

X_test = pd.get_dummies(X_test,columns=['Sex','Age','Fare'],dummy_na=False)
X_test.describe()
titanic_test['Survived'] = best_dt_estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\usunkesu\\Desktop\\submission.csv", columns=["PassengerId", "Survived"], index=False)

#Kaggle Submission Score
#Your Best Entry 
#Your submission scored 0.78468





