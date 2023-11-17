import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_csv("C:/Users/krish/Documents/Sem 5 NCU/clg AIML/bottle.csv")

# Print dataset information
print("Dataset Shape:", dataset.shape)
print("Dataset Info:")
print(dataset.info())

# Select specific columns (temperature and salinity)
df = dataset[['T_degC', 'Salnty']]

# Handle missing values
df.isna().sum()
df.dropna(axis=0, inplace=True)

# Sample only the first 1000 rows
df = df.sample(1000)

# Summary statistics of the selected columns
print("Dataframe Shape:", df.shape)
print("Dataframe Describe:")
print(df.describe())

# Data visualization
plt.scatter(x=df['T_degC'], y=df['Salnty'], alpha=0.8)
plt.xlabel("Water Temperature")
plt.ylabel("Water Salinity")
plt.show()

# Prepare data for linear regression
X = df[['T_degC']]
y = df['Salnty']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)

# Create and fit a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Model coefficients
m = regressor.coef_
b = regressor.intercept_
print("Slope (m):", m)
print("Intercept (b):", b)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print("R-squared (R2) Score:", r2)

# Generate input data for the regression line
x_input = np.linspace(0, 30, 500)
y_input = m * x_input + b

# Data visualization with regression line
plt.scatter(x=X_train, y=y_train, alpha=0.8)
plt.plot(x_input, y_input, c='r')
plt.xlabel('Water Temperature')
plt.ylabel('Water Salinity')
plt.show()