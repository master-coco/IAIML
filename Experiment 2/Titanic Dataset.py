#Titanic Challenge: The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, 
#during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. 
#Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew. 
#While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. 
#In this project, the students need to build a predictive model that answers the question: 
#“what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/krish/Documents/Sem 5 NCU/clg AIML/Experiment 2/Titanic-Dataset.csv")

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print(df.shape)

print(df.sample(5))

df = df.drop(columns = ['PassengerId' , 'Name' , 'Ticket' , 'Cabin'])

print(df.head(5))
print(df.shape)

print(df.isnull().sum())
print(df.describe())

print(df['Embarked'].value_counts())
print(df['Sex'].value_counts())

sns.pairplot(df[['Age','SibSp','Parch','Fare']])

plt.show()

X = df.iloc[:,1:]
print(X.shape)

y = df.iloc[:,:1]
print(y.shape)

X_train , X_test , y_train , y_test = train_test_split(X,y , random_state = 0 , test_size = 0.3)