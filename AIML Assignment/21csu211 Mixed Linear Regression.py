# Assignment for AIML course, 

# At the following google drive link 
# https://drive.google.com/drive/u/1/folders/1A-Ht7qygiKVeKEC8RJgsCk5c4VMjMPsX 
# Create a folder by your roll number and Submit .ipynb/.py files and the dataset in that folder on any one of the following regression methods

# 1 mixed linear regression 
# 2 dmine regression
# 3 Deming regression
# 4 Passing-bablok regression

# Deadline: 8 Oct 11.59 PM

#Arjun Bhardwaj 21csu211

import numpy as np
import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the diabetes dataset from scikit-learn
diabetes = datasets.load_diabetes()
X = diabetes.data[:, 2].reshape(-1, 1)  # Use a single feature
y = diabetes.target

# Simulate a group variable (for mixed linear regression)
np.random.seed(0)
group_variable = np.random.randint(1, 6, size=len(y))

# Simple Linear Regression
#simple_model = sm.OLS(y, sm.add_constant(X)).fit()

# Mixed Linear Regression (Random Intercept Model)
data = pd.DataFrame({'X': X[:, 0], 'y': y, 'Group': group_variable})
mixed_model = sm.MixedLM.from_formula("y ~ X", data=data, groups=data["Group"])
mixed_result = mixed_model.fit()

# Mixed Linear Regression Plot
plt.subplot(1, 2, 2)
for group in data['Group'].unique():
    group_data = data[data['Group'] == group]
    plt.scatter(group_data['X'], group_data['y'], label=f"Group {group}")
plt.plot(data['X'], mixed_result.predict(data), color='red', label="Mixed Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Mixed Linear Regression")
plt.legend()

plt.tight_layout()
plt.show()