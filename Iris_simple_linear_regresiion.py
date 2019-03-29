# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:01:44 2019

@author: Jostin Joseph
"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv('https://s3.eu-geo.objectstorage.softlayer.net/ml-datasetstore/iris.csv',names=names)

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import OneHotEncoder
onehotencoder= OneHotEncoder()

y=onehotencoder.fit_transform(y)

 #create training and test set
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder=LabelEncoder()

y=labelencoder.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=42)


#create Linear Regression

from sklearn.linear_model import LinearRegression

linearregression= LinearRegression()

linearregression.fit(X_train,y_train)

y_pred=linearregression.predict(X_test)

# CHeck the correct predictions
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)