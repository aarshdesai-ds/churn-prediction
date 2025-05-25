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

> A fully interactive local app to predict customer churn probability based on user input.

### 🧾 Features
- Sidebar with inputs like gender, contract type, billing preferences
- Real-time churn prediction with confidence score
- Clean, responsive design running locally

---

## 🚀 How to Run Locally

1. Clone the repository
`git clone https://github.com/aarshdesai-ds/churn-prediction.git`
`cd churn-prediction`

2. Install dependencies
`pip install -r requirements.txt`

3. Run the Streamlit app
`streamlit run app/streamlit_app.py`

---

## ✅ Key Takeaways

This project combines a machine learning pipeline with a Streamlit web application to predict telecom customer churn. Here are the major insights and results:

---

## 🔍 Real-World Problem, Real-World Features

- Used rich customer data including **demographics**, **services used**, and **billing information** to predict churn behavior.
- Feature engineering and encoding enabled better understanding of the customer lifecycle and engagement levels.

---

## ⚖️ Class Imbalance Solved with SMOTE

- The original dataset was **imbalanced**, with more non-churn than churn cases.
- **SMOTE (Synthetic Minority Over-sampling Technique)** improved recall for the minority churn class.
- Helped the model detect more true churn cases without overfitting.

---

## 🌲 Robust Model Performance

- **Random Forest Classifier** provided strong performance with:
  - **85% Accuracy**
  - **84.1% Precision**
  - **86.8% Recall**
  - **ROC-AUC > 0.85**
- Balanced performance meant it could catch churners **without overwhelming false positives**.

---

## 🧪 Business-Relevant Evaluation

- **High Recall** ensured actual churners are rarely missed — crucial for proactive retention efforts.
- **Precision** kept false churn alerts low — important to avoid unnecessary interventions for loyal users.
- **Confusion Matrix** illustrated a strong separation between classes.

---

## 🖥️ Interactive Streamlit App

- Fully local, interactive app for customer churn prediction  
- Users can input customer attributes and get **real-time probability predictions**  
- Clean UI supports **operational deployment** or internal demos  

---

## 📈 Conclusion

This churn prediction system is both **technically robust** and **business-friendly**.  
With strong recall, balanced metrics, and an interactive front-end, it’s ideal for customer success teams aiming to **identify at-risk users early and take action**.
