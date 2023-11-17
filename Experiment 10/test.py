import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

wine = load_wine()

X_wine = wine.data[:, :4]
y_wine = wine.target

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

k = 3  
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train_iris, y_train_iris)

y_pred_wine = knn.predict(X_test_wine)

accuracy_wine = accuracy_score(y_test_wine, y_pred_wine)
print(f'Accuracy on Wine dataset : {accuracy_wine*100:.2f}%')