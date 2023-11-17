import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate a synthetic dataset for Polynomial Regression
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=0)

# Create polynomial features with degree=5 for much more curve          //no labels
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

# Fit a linear regression model to the polynomial features         //dont use linear regression(pass degree in parameter)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Create a range of X values for plotting
X_range = np.arange(X.min(), X.max(), 0.1)[:, np.newaxis]
X_range_poly = poly.transform(X_range)

# Predict the y values using the polynomial regression model
y_range_poly = poly_reg.predict(X_range_poly)

# Plot the data points and the polynomial regression line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_range, y_range_poly, color='red', label='Polynomial Regression (Degree=5)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression with Much More Curve (Degree=5)')
plt.show()