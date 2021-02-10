#dataset : https://archive.ics.uci.edu/ml/datasets/wine+quality
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


wine_data = pd.read_csv('winequailty-red.csv', sep = ';')
print(wine_data.columns.tolist())

X = wine_data.drop('quality', axis = 1).values
y = wine_data['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_test_splitze = 0.1, random_state = 42)


reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(y_pred)
print(y_test)

score = reg.score(X_test, y_test)
print(score)
