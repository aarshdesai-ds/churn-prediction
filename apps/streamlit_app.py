# streamlit_app.py

# 📉 Customer Churn Predictor App
# Uses a trained Random Forest model to predict the likelihood of churn based on customer inputs

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load the trained model ===
model = joblib.load('churn_model.pkl')


# === App Title ===
st.title("📉 Customer Churn Predictor")

st.markdown("""
This Streamlit app uses a machine learning model to predict whether a customer is likely to **churn**.
Fill in the customer details on the left, and click 'Predict Churn' to get a real-time prediction.
""")

# === Sidebar: Input customer data ===
st.sidebar.header("🧾 Customer Profile")

gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ['Yes', 'No'])
Partner = st.sidebar.selectbox("Has Partner?", ['Yes', 'No'])
Dependents = st.sidebar.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.sidebar.selectbox("Phone Service", ['Yes', 'No'])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])
MonthlyCharges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
TotalCharges = st.sidebar.slider("Total Charges", 0, 9000, 2000)
Contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
InternetService = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])

# === Feature Engineering: Convert inputs into model-ready format ===
input_dict = {
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0,
    'Partner': 1 if Partner == 'Yes' else 0,
    'Dependents': 1 if Dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if PhoneService == 'Yes' else 0,
    'PaperlessBilling': 1 if PaperlessBilling == 'Yes' else 0,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,

    # One-hot encoded categorical features
    'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
    'InternetService_No': 1 if InternetService == 'No' else 0,
    'Contract_One year': 1 if Contract == 'One year' else 0,
    'Contract_Two year': 1 if Contract == 'Two year' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0
}

# Ensure all expected features are present (fill missing with 0)
expected_features = model.feature_names_in_
for feature in expected_features:
    if feature not in input_dict:
        input_dict[feature] = 0

# Convert to DataFrame and match feature order
input_df = pd.DataFrame([input_dict])[model.feature_names_in_]

# === Predict ===
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn! \n\n**Churn Probability:** {probability:.2%}")
    else:
        st.success(f"✅ This customer is not likely to churn. \n\n**Churn Probability:** {probability:.2%}")
