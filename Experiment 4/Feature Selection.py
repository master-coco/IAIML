import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load a sample dataset (Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Create a DataFrame for the dataset
df = pd.DataFrame(data=np.c_[X, y], columns=data.feature_names + ['target'])
print(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply feature selection techniques
print("Original number of features:", X_train.shape[1])

# Filter method 1: SelectKBest using chi-squared test
select_chi2 = SelectKBest(score_func=chi2, k=2)
X_train_chi2 = select_chi2.fit_transform(X_train, y_train)
X_test_chi2 = select_chi2.transform(X_test)
print("Number of features selected by chi-squared test:", X_train_chi2.shape[1])

# Filter method 2: SelectKBest using F-statistic (ANOVA)
select_f = SelectKBest(score_func=f_classif, k=2)
X_train_f = select_f.fit_transform(X_train, y_train)
X_test_f = select_f.transform(X_test)
print("Number of features selected by F-statistic (ANOVA):", X_train_f.shape[1])

# Filter method 3: SelectKBest using mutual information
select_mi = SelectKBest(score_func=mutual_info_classif, k=2)
X_train_mi = select_mi.fit_transform(X_train, y_train)
X_test_mi = select_mi.transform(X_test)
print("Number of features selected by mutual information:", X_train_mi.shape[1])

# Train a classifier (e.g., RandomForest) and measure accuracy
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred)

classifier_chi2 = RandomForestClassifier(random_state=42)
classifier_chi2.fit(X_train_chi2, y_train)
y_pred_chi2 = classifier_chi2.predict(X_test_chi2)
accuracy_chi2 = accuracy_score(y_test, y_pred_chi2)

classifier_f = RandomForestClassifier(random_state=42)
classifier_f.fit(X_train_f, y_train)
y_pred_f = classifier_f.predict(X_test_f)
accuracy_f = accuracy_score(y_test, y_pred_f)

classifier_mi = RandomForestClassifier(random_state=42)
classifier_mi.fit(X_train_mi, y_train)
y_pred_mi = classifier_mi.predict(X_test_mi)
accuracy_mi = accuracy_score(y_test, y_pred_mi)

print(f"Accuracy (Original): {accuracy_original}")
print(f"Accuracy (Chi-squared): {accuracy_chi2}")
print(f"Accuracy (F-statistic): {accuracy_f}")
print(f"Accuracy (Mutual Information): {accuracy_mi}")