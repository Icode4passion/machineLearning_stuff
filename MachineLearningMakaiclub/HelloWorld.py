# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:23:24 2017

@author: usunkesu
"""

from sklearn import tree
features =[[100,0],[110,0],[120,0],[130,1],[125,1]]
labels = [1,1,1,0,0]
t = tree.DecisionTreeClassifier()
t.fit(features,labels)
p = t.predict([115,1])
print (p)