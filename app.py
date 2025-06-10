import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model("model.json")

# Feature explanations dictionary (for user-friendly reasons)
feature_reason_map = {
    "Credit_History": {
        0: "No credit history available",
        1: "Good credit history"
    },
    "TotalIncome": "Low total income",
    "LoanAmount": "Requested loan amount is high",
    "Property_Area_Semiurban": "Semiurban property may impact eligibility",
    "Education_Not Graduate": "Not being a graduate may reduce approval chances",
}

st.set_page_config(page_title="Loan Eligibility Prediction", layout="wide")
st.title("\U0001F3E6 Loan Eligibility Prediction with Explainable AI")

st.write("Fill in applicant details to check loan eligibility and explanation.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.number_input("Loan Term (in days)", value=360)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Combine income fields
total_income = applicant_income + coapplicant_income

# Prepare input features as per training format
input_dict = {
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'TotalIncome': [total_income],
    'Gender_Female': [1 if gender == 'Female' else 0],
    'Gender_Male': [1 if gender == 'Male' else 0],
    'Married_No': [1 if married == 'No' else 0],
    'Married_Yes': [1 if married == 'Yes' else 0],
    'Dependents_0': [1 if dependents == '0' else 0],
    'Dependents_1': [1 if dependents == '1' else 0],
    'Dependents_2': [1 if dependents == '2' else 0],
    'Dependents_3+': [1 if dependents == '3+' else 0],
    'Education_Graduate': [1 if education == 'Graduate' else 0],
    'Education_Not Graduate': [1 if education == 'Not Graduate' else 0],
    'Self_Employed_No': [1 if self_employed == 'No' else 0],
    'Self_Employed_Yes': [1 if self_employed == 'Yes' else 0],
    'Property_Area_Rural': [1 if property_area == 'Rural' else 0],
    'Property_Area_Semiurban': [1 if property_area == 'Semiurban' else 0],
    'Property_Area_Urban': [1 if property_area == 'Urban' else 0],
}
input_df = pd.DataFrame(input_dict)

# Predict and explain
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_text = "\u2705 Loan Approved" if prediction == 1 else "\u274C Loan Rejected"
    st.subheader(f"Prediction: {prediction_text}")

    # SHAP explanation using safe plotting
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    # Extract top contributing features
    shap_impact = shap_values.values[0]
    feature_names = input_df.columns
    top_features = sorted(zip(feature_names, shap_impact), key=lambda x: abs(x[1]), reverse=True)[:3]

    # User-friendly explanation
    st.subheader("Top Reasons:")
    for feat, val in top_features:
        reason = ""
        if feat == "Credit_History":
            reason = feature_reason_map[feat].get(int(input_df[feat][0]), "")
        elif feat in feature_reason_map:
            reason = feature_reason_map[feat]
        if reason:
            st.write(f"\u2022 {reason} ({'+' if val > 0 else ''}{round(val, 2)})")

    # Optional SHAP bar chart (simplified)
    with st.expander("Show SHAP Feature Impact Chart"):
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
