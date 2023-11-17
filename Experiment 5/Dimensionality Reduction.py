#Exp 5 Study Dimensionality Reduction.

# Understand the basic principle behind Principal Component Analysis.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Create a DataFrame to describe the dataset
iris_df = pd.DataFrame(X, columns=data.feature_names)
iris_df['Target'] = y

# Use the describe() method to generate summary statistics
summary_statistics = iris_df.describe()

print(iris_df.info())
print(iris_df.head())

# Display the summary statistics
#print(summary_statistics)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)  # You can choose the number of components you want to keep
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.colorbar(label='Target Class')
plt.show()

explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratios:", explained_variance)