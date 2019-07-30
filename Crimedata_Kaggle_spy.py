# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 19:20:15 2019

@author: usunkesu
"""

#tutorial from 
'''
https://www.kaggle.com/pranoybiswas/crime-data-eda
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\usunkesu\\Documents\\Mahesh\\ML\\Victims_of_rape.csv")
#df.head()
df.info()
df_ap = df.loc[df['Area_Name']=='Andhra Pradesh']
df_ap.head()

ax = df_ap[['Victims_Between_10-14_Yrs','Victims_Between_14-18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-50_Yrs']].plot(kind='bar')
ax.set_xticklabels([''])
plt.show()

# from pandas.plotting import scatter_matrix
# df_ap_drop = df_ap.drop(['Area_Name','Subgroup'],axis=1)
# scatter_matrix(df_ap_drop)
# plt.show()

df_ap_cat = df_ap.select_dtypes(include=['object'])
df_ap_cat.head()

df_ap_pivot = pd.pivot_table(df_ap,values=('Victims_Between_10-14_Yrs','Victims_Between_14-18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-50_Yrs'),index=['Year','Subgroup'])

df_ap_pivot.uns