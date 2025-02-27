import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from src.recommendation import get_recommendation

# Load the trained model
model = pickle.load(open("models/diabetes_risk_model.pkl", "rb"))

# Load the scaler
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Streamlit UI
st.title("🩺 MediAI: Disease Risk Assessment")
st.write("Enter your health parameters to assess your risk level.")

# Collect user input
pregnancies = st.number_input("Number of Pregnancies (0 if not applicable):", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level:", min_value=50, max_value=250)
blood_pressure = st.number_input("Blood Pressure:", min_value=50, max_value=200)
skin_thickness = st.number_input("Skin Thickness:", min_value=0, max_value=100)
insulin = st.number_input("Insulin Level:", min_value=0, max_value=900)
bmi = st.number_input("Body Mass Index (BMI):", min_value=10.0, max_value=60.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Enter your Age:", min_value=1, max_value=120, step=1)

# Predict risk level when button is clicked
if st.button("Check Risk Level"):
    # Create input array in the correct format
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Apply the same scaling used in training
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    predicted_risk = model.predict(input_data_scaled)[0]
    recommendation = get_recommendation(predicted_risk)

    # Display results
    st.subheader(f"Predicted Risk Level: {predicted_risk}")
    st.success(f"Health Recommendation: {recommendation}")
