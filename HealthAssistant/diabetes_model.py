import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
df = pd.read_csv("datasets/diabetes.csv")

# Handle zero values in key columns
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())

feature_names = df.drop(columns=["Outcome"]).columns.tolist()
X = df[feature_names].values
y = df["Outcome"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# Train logistic regression
best_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
best_clf.fit(X_train, y_train)

# Predictions
y_train_pred = best_clf.predict(X_train)
y_test_pred = best_clf.predict(X_test)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print("=======================================================")
print("Diabetes Prediction Model Evaluation")
print("=======================================================")
print("Confusion Matrix (Test):\n", cm)
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print("=======================================================")

# Accuracy visualization
plt.bar(["Train", "Test"], [train_acc, test_acc], color=["lightblue", "lightcoral"])
plt.title("Diabetes Model - Train vs Test Accuracy")
plt.ylabel("Accuracy")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Diabetes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()





