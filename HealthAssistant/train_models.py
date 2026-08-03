#!/usr/bin/env python3
"""
Train all three models once and save bundled .sav files
Each bundle is a dict: { 'model': <sklearn estimator>, 'scaler': <sklearn scaler>, 'features': [feature names] }
Saved to: HealthAssistant/models/<disease>_model.sav
"""
from pathlib import Path
import pickle
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "datasets"
MODELS_DIR = BASE_DIR / "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_save_diabetes():
    df = pd.read_csv(DATA_DIR / "diabetes.csv")
    # Handle zero values in certain columns (as original notebook did)
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zeros:
        if col in df.columns:
            df[col] = df[col].replace(0, df[col].median())

    feature_names = df.drop(columns=["Outcome"]).columns.tolist()
    X = df[feature_names].values
    y = df["Outcome"].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    # Evaluation (prints)
    y_test_pred = model.predict(X_test)
    print("Diabetes model test accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    bundle = {"model": model, "scaler": scaler, "features": feature_names}
    path = MODELS_DIR / "diabetes_model.sav"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print("Saved:", path)


def train_and_save_heart():
    df = pd.read_csv(DATA_DIR / "heart.csv")
    feature_names = df.drop(columns=["target"]).columns.tolist()
    X = df[feature_names].values
    y = df["target"].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    print("Heart model test accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    bundle = {"model": model, "scaler": scaler, "features": feature_names}
    path = MODELS_DIR / "heart_model.sav"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print("Saved:", path)


def train_and_save_parkinsons():
    df = pd.read_csv(DATA_DIR / "parkinsons.csv")
    if "name" in df.columns:
        df = df.drop(columns=["name"])

    feature_names = df.drop(columns=["status"]).columns.tolist()
    X = df[feature_names].values
    y = df["status"].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    print("Parkinsons model test accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    bundle = {"model": model, "scaler": scaler, "features": feature_names}
    path = MODELS_DIR / "parkinsons_model.sav"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print("Saved:", path)


def main():
    print("Training Diabetes model...")
    train_and_save_diabetes()
    print("\nTraining Heart model...")
    train_and_save_heart()
    print("\nTraining Parkinson's model...")
    train_and_save_parkinsons()
    print("\nAll models trained and saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()
