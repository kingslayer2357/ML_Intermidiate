# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:31:18 2020

@author: kingslayer
"""

#RANDOM FOREST REGRESSION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv(r"Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values

#creating and random forest model and fitting to dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(X,y)

#plotting
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="green")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("SALARY VS POSITION")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

#predicting 
y_pred=regressor.predict([[6.5]])
