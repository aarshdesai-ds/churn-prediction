# churn_eda_modeling.py

# 📊 Customer Churn Prediction - EDA, Preprocessing, and Modeling Script
# Author: Aarsh Desai
# Goal: Prepare a clean dataset, handle imbalance, train a Random Forest model, and save it for deployment

# === Import Required Libraries ===
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# === Load Dataset ===
data_path = '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data_path)

# === Preview Dataset ===
print("🔍 First 5 rows:")
print(df.head())
print("\n📋 Data Info:")
print(df.info())

# === Drop Unnecessary Columns ===
# 'customerID' is just an identifier — not predictive
df.drop('customerID', axis=1, inplace=True)

# === Convert TotalCharges to Numeric ===
# Some values are strings (blanks), so we'll convert them and coerce errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# === Handle Missing Values ===
# Fill missing TotalCharges with median (a robust measure)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# === Encode Binary Categorical Variables ===
# Replace Yes/No and Male/Female with 1/0
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})

# === One-Hot Encode Multi-Category Columns ===
# Convert string categories into numerical columns using one-hot encoding
df = pd.get_dummies(df, columns=[
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
], drop_first=True)

# === Feature Scaling ===
# Standardize numerical features so they are on the same scale
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df[['tenure', 'MonthlyCharges', 'TotalCharges']]
)

# === Split Features and Target ===
X = df.drop('Churn', axis=1)
y = df['Churn']

# === Handle Imbalanced Dataset with SMOTE ===
# Since 'Churn' is imbalanced, use SMOTE to create synthetic minority samples
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# === Split into Training and Testing Sets ===
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# === Train a Random Forest Classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate Model Performance ===
y_pred = model.predict(X_test)

print("\n🧪 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🎯 ROC-AUC Score:")
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# === Save the Trained Model ===
# Create the model directory if it doesn’t exist
os.makedirs('../model', exist_ok=True)
joblib.dump(model, '../model/churn_model.pkl')

print("\n✅ Model saved successfully to 'model/churn_model.pkl'")

