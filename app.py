import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit UI
st.title("Insurance Claim Prediction App")
st.write("Enter the details below to predict whether an insurance claim will be made.")

# Input Fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
steps = st.number_input("Steps Per Day", min_value=1000, max_value=30000, value=5000)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
charges = st.number_input("Medical Charges", min_value=1000, max_value=100000, value=10000)
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

# Encode categorical inputs
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
region_mapping = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}
region = region_mapping[region]

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, bmi, steps, children, charges, sex, smoker, region]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = "Will Claim Insurance" if prediction == 1 else "No Insurance Claim"
    st.success(f"Prediction: {result}")
