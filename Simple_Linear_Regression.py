# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fit SLR to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test results
y_pred = regressor.predict(x_test)

# Visualizing the Test set results
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience'); plt.ylabel('Salary')
plt.show()

# Visualizing the training set results
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience'); plt.ylabel('Salary')
plt.show()