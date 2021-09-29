import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

dataset = pd.read_csv("Iris.csv")

#Splitting the data into training and test test
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values
print(Y)
X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)