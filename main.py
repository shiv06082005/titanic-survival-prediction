
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create output folder
os.makedirs("output", exist_ok=True)

import os
import urllib.request

# Create data folder
os.makedirs("data", exist_ok=True)

# Dataset URL (Titanic dataset)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
file_path = "data/train.csv"

# Download if not exists
if not os.path.exists(file_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, file_path)
    print("Download complete!")

# Load dataset
df = pd.read_csv(file_path)

# Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# EDA plots
plt.figure(figsize=(10,5))
sns.countplot(x='Survived', data=df)
plt.savefig("output/eda_plots.png")
plt.close()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.savefig("output/correlation_heatmap.png")
plt.close()

# Model
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.savefig("output/confusion_matrix.png")
plt.close()

# Feature importance
importance = model.coef_[0]
sns.barplot(x=importance, y=features)
plt.savefig("output/feature_importance.png")
plt.close()

print("Accuracy:", accuracy_score(y_test, y_pred))
