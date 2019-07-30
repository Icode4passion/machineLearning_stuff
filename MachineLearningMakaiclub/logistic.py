# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 23:16:53 2017

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


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test =sc.fit_transform(X_test)


from sklearn.linear_model import LogisticRegression
c = LogisticRegression()
c.fit(X_train,y_train)


y_pred =c.predict(X_test)
print(y_pred)
