# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:23:28 2023

@author: Dell
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\Data Science\Daily Practice\March\17-03-2023\EMP SAL.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.neighbors import KNeighborsRegressor
regressor= KNeighborsRegressor(n_neighbors=3,metric='minkowski')
regressor.fit(x,y)

from sklearn.svm import SVR

regressor=SVR(kernel="poly", degree=5, gamma="auto")
regressor.fit(x,y)

y_pred=regressor.predict([[6.5]])
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
