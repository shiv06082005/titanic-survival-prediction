import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create folders
os.makedirs("output", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Download dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
file_path = "data/train.csv"

if not os.path.exists(file_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, file_path)
    print("Download complete!")

# Load data
df = pd.read_csv(file_path)

# Preprocessing
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# ------------------ EDA PLOTS ------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.countplot(x='Survived', data=df)

plt.subplot(1,2,2)
sns.countplot(x='Sex', hue='Survived', data=df)

plt.tight_layout()
plt.savefig("output/eda_plots.png")
plt.close()

# ------------------ HEATMAP ------------------
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.savefig("output/correlation_heatmap.png")
plt.close()

# ------------------ MODEL ------------------
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ------------------ CONFUSION MATRIX ------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("output/confusion_matrix.png")
plt.close()

# ------------------ FEATURE IMPORTANCE ------------------
importance = model.coef_[0]

plt.figure(figsize=(8,5))
sns.barplot(x=importance, y=features)
plt.savefig("output/feature_importance.png")
plt.close()

# ------------------ RESULT ------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
