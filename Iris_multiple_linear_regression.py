# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:01:44 2019

@author: josti
"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv('iris.csv',names=names)

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import OneHotEncoder
onehotencoder= OneHotEncoder()


y=onehotencoder.fit_transform(y)

    print(y)

 #create training and test set
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder=LabelEncoder()

y=labelencoder.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=42)



#create Multiple Linear Regression

from sklearn.linear_model import LinearRegression

linearregression= LinearRegression()

linearregression.fit(X_train,y_train)

y_pred=linearregression.predict(X_test)



import statsmodels.formula.api as sm

X=np.append(arr=np.ones((150,1)).astype(int),values=X,axis=1)

X_opt=X[:, [0,1,2,3,4]]


regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()

regressor_OLS.summary()

X_opt=X[:, [0,1,3,4]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()

X_opt=X[:, [1,3,4]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()


