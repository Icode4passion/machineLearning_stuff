# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:39:00 2017

@author: usunkesu
"""
#Basic Requirements
import numpy as np
import matplotlib.pyplot as mat
import pandas as pd

#Where's my Data?
data = pd.read_csv('winequality-red.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, 11]

#Ninja needs to be trined and tested
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Train the Ninja
from sklearn.linear_model import LinearRegression
r = LinearRegression()
r.fit(X_train,y_train)

#Test the Ninja
lr_y_pred = r.predict(X_test)
print(lr_y_pred)

#Rounding up the predicted values
for i in range(len(lr_y_pred)):
    lr_y_pred[i] = round(lr_y_pred[i])
print(lr_y_pred)



#Evaluate the Results
def rightCheck(y_test2, lr_y_pred):
    got_right = 0
    for i in range(len(y_test)):
        if(y_test2[i]==lr_y_pred[i]):
            got_right = got_right + 1
    return got_right
            
print("Total Length: ", len(y_test))
print("Total right Predictions: ", rightCheck(y_test.values, lr_y_pred))



#Tuning the Ninja Moves by Backward Elimination
import statsmodels.formula.api as sm
#Tuning Step 1
X = np.append(arr = np.ones((len(data), 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
r_OLS = sm.OLS(endog = y, exog = X_opt).fit()
r_OLS.summary()