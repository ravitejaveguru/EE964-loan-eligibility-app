import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = xgb.XGBClassifier()
model.load_model("model.json")

st.set_page_config(page_title="Loan Eligibility Prediction", layout="wide")
st.title("🏦 Loan Eligibility Prediction with Explainable AI")

st.write("Fill in applicant details to check loan eligibility and explanation.")

# Input fields
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

# Feature engineering
total_income = applicant_income + coapplicant_income

# Create dataframe for prediction
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

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_text = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    st.subheader(f"Prediction: {prediction_text}")

    # SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Feature Impact Explanation:")
    shap.plots.waterfall(shap_values[0])
    st.pyplot()
