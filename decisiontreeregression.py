# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:21:32 2020

@author: kingslayer
"""
#DECISION TREE REGRESSION

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
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X=sc_x.fit_transform(X)
y=sc_y.fit_transform(y)"""

#creating and fitting Decision Tree model
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#plotting graph
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("SALARY VS POSITION(SVR)")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

#predicting
y_pred=regressor.predict([[6.5]])

#for real and higher resolution
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("SALARY VS POSITION(SVR)")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()
