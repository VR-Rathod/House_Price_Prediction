#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy pandas matplotlib scikit-learn


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the California Housing dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Use a single feature (e.g., average number of rooms per dwelling) for simplicity
X = data[['MedInc']]  # Feature: median income
y = data['PRICE']  # Target: house prices

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Visualize the regression line and the data points
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Median Income')
plt.ylabel('House Price')
plt.title('Linear Regression: House Price Prediction')
plt.show()


# In[ ]:




