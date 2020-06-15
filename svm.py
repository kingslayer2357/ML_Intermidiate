# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:29:06 2020

@author: kingslayer
"""

#SVR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv(r"Position_Salaries.csv")

X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values

#splitting into training and test set
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X=sc_x.fit_transform(X)
y=sc_y.fit_transform(y)

#creating and fitting SVR Model
from sklearn.svm import SVR
regressor=SVR(kernel="rbf")
regressor.fit(X,y)

#plotting graph
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("SALARY VS POSITION(SVR)")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

#predicting
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
