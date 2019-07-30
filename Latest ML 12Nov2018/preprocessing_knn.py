# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:34:09 2018

@author: usunkesu
"""
'''
tutorial reference from 
https://www.datacamp.com/community/tutorials/preprocessing-in-data-science-part-1-centering-scaling-and-knn
Preprocessing in Data Science (Part 1): Centering, Scaling, and KNN

'''




import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import neighbors , linear_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale



plt.style.use('ggplot')
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',sep=';')
X = df.drop('quality', 1).values
y1 = df['quality'].values   
pd.DataFrame.hist(df, figsize=[15,15])


y = y1<=5
plt.figure(figsize=(20,5))
plt.hist(y1);
plt.xlabel('original target value')
plt.ylabel('count')
plt.subplot(1, 2, 2);
plt.hist(y)
plt.xlabel('aggregated target value')
plt.show()


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2 , random_state= 42)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_model_one = knn.fit(X_train,y_train)
print("knn accuracy for test set %f" % knn_model_one.score(X_train,y_train))
y_true , y_pred = y_test , knn_model_one.predict(X_test)
print(classification_report(y_true,y_pred))


#The mechanics of preprocessing: scaling and centering



Xs = scale(X)
Xs_train , Xs_test , y_train , y_test = train_test_split(Xs,y,test_size=0.2,random_state=42)
knn_model_two = knn.fit(Xs_train,y_train)
print("Knn score for test set %f" %knn_model_two.score(Xs_test,y_test))
print("Knn score for train set %f" %knn_model_two.score(Xs_train,y_train))
y_true , y_pred = y_test , knn_model_two.predict(Xs_test)
print(classification_report(y_true , y_pred))





















