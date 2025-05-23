# 📉 Customer Churn Prediction

This project predicts telecom customer churn using a machine learning pipeline and an interactive **Streamlit web app**.  
It combines feature engineering, class imbalance handling, and model deployment to provide actionable business insights.

---

## 📦 Dataset

- **Source**: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Target**: `Churn` (Yes/No)  
- **Features**:
  - Customer demographics: Gender, SeniorCitizen, Partner, Dependents
  - Account info: Tenure, Contract type, Billing preferences
  - Services used: Phone, Internet, Streaming, Security, Support
  - Charges: MonthlyCharges, TotalCharges

---

## 🧠 ML Pipeline Overview

### 🧹 Data Preparation
- Converted `TotalCharges` to numeric and imputed missing values
- One-hot encoded multi-category features (e.g., Contract, InternetService)
- Scaled numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`)

### ⚖️ Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance churn classes (Yes/No)

### 🌲 Modeling
- Trained a **Random Forest Classifier** with 100 trees
- Evaluated using metrics like **Accuracy**, **Precision**, **Recall**, and **ROC-AUC**

---

## 🎯 Model Evaluation

### 🧪 Confusion Matrix

|                      | Predicted: No Churn  | Predicted: Churn     |
| -------------------- | -------------------- | -------------------- |
| **Actual: No Churn** | 849 (True Negative)  | 172 (False Positive) |
| **Actual: Churn**    | 138 (False Negative) | 911 (True Positive)  |


### 📊 Metrics

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | 85.0%     |
| Precision     | 84.1%     |
| Recall        | 86.8%     |
| ROC-AUC       | 0.85+     |

**Interpretation**:
- High **Recall** ensures most actual churners are caught — ideal for business intervention
- **Precision** ensures we don’t falsely alert too many loyal customers
- Balanced model suitable for customer retention systems

---

## 💻 Streamlit Web App

> A fully interactive app to predict customer churn probability based on live inputs.

### 🧾 Features
- Sidebar with inputs like gender, contract type, billing preferences
- Live churn prediction with confidence score
- Clean, responsive design

### ▶️ Try it Online

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)

> 📌 Replace `YOUR-APP-URL` with your deployed app URL on [Streamlit Cloud](https://streamlit.io/cloud)

---


## 🚀 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/streamlit_app.py


