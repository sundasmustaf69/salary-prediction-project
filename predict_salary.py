#  To run this app, open terminal and type:
#    streamlit run predict_salary.py
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# --- Load Model and Scaler ---
model = load_model('salary_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- App Title ---
st.set_page_config(page_title="Salary Prediction App", page_icon="")
st.title(" Salary Prediction App")
st.write("Fill the following details to estimate the customer's salary.")

# --- Input Fields ---
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years with bank)", min_value=0, max_value=20, value=5)
balance = st.number_input("Account Balance", min_value=0.0, value=10000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
prev_est_salary = st.number_input("Previous Estimated Salary", min_value=0.0, value=50000.0)
gender = st.radio("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# --- Encoding ---
gender = 1 if gender == "Male" else 0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# --- Prepare Features ---
features = np.array([[credit_score, gender, age, tenure, balance,
                      num_of_products, has_cr_card, is_active_member,
                      prev_est_salary, geo_germany, geo_spain]])

features_scaled = scaler.transform(features)

# --- Prediction ---
if st.button("Predict Salary"):
    predicted_salary = model.predict(features_scaled)[0][0]
    st.success(f"💸 Predicted Salary: Rs {predicted_salary:,.2f}")







