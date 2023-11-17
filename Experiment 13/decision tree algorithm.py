# Consider iris data set or diabetic data set and classify the problem using decision tree algorithm by using following libraries 

#for encoding
from sklearn.preprocessing import LabelEncoder 

#for train test splitting
from sklearn.model_selection import train_test_split

#for decision tree object
from sklearn.tree import DecisionTreeClassifier

#for checking testing results
from sklearn.metrics import classification_report, confusion_matrix

#for visualizing tree 
from sklearn.tree import plot_tree

#decision tree
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X, y)

class_names = iris.target_names.tolist()

# Visualize the decision tree
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=class_names, rounded=True)
plt.show()

# print accuracy (not 1) if 1 check again