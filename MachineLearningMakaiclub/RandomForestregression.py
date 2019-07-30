# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:48:29 2017

@author: usunkesu
"""

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


from sklearn.ensemble import RandomForestRegressor

r = RandomForestRegressor(n_estimators = 10, random_state=0)
r.fit(X_train,y_train)
y_pred = r.predict(X_test)

for i in range(len(y_pred)):
    y_pred[i] = round(y_pred[i])
    
    
