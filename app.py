import streamlit as st
import numpy as np
import joblib

# Load saved components
model = joblib.load("best_diabetes_model.pkl")
pca = joblib.load("pca_transformer.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient medical details to predict Diabetes status.")

# Input fields
Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose Level", 0, 300, 120)
BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
Insulin = st.number_input("Insulin Level", 0, 900, 80)
BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
Age = st.number_input("Age", 1, 120, 30)

# Prediction Button
if st.button("🔍 Predict"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                             Insulin, BMI, DPF, Age]])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # PCA Transform
    input_pca = pca.transform(input_scaled)

    # Prediction
    prediction = model.predict(input_pca)[0]

    if prediction == 1:
        st.error("⚠️ The patient is **Diabetic**")
    else:
        st.success("✅ The patient is **Non-Diabetic**")
