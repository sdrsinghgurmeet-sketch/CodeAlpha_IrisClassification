import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris


iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target

print(df.head())


if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Encode species labels
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Try different models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

#  Compare performance
plt.bar(results.keys(), results.values(), color=['blue','green','orange','red'])
plt.ylabel("Accuracy")
plt.title("Model Comparison on Iris Dataset")
plt.show()

# Pairplot visualization
sns.pairplot(df, hue="Species", diag_kind="kde", palette="husl")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()


# : Confusion Matrix for best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


features = df.columns[:-1]  # all columns except 'Species'

for feature in features:
    plt.figure(figsize=(6,4))
    sns.histplot(data=df, x=feature, hue="Species", kde=True, palette="husl")
    plt.title(f"Distribution of {feature}")
    plt.show()


plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
