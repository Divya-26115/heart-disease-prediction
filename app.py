import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("❤️ Heart Disease Prediction System")

# Input fields
age = st.number_input("Age")
sex = st.number_input("Sex (1=Male, 0=Female)")
cp = st.number_input("Chest Pain Type")
trestbps = st.number_input("Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.number_input("Fasting Blood Sugar")
restecg = st.number_input("ECG")
thalach = st.number_input("Max Heart Rate")
exang = st.number_input("Exercise Induced Angina")
oldpeak = st.number_input("Oldpeak")
slope = st.number_input("Slope")
ca = st.number_input("CA")
thal = st.number_input("Thal")

# Prediction
if st.button("Predict"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]])

    result = model.predict(data)

    if result[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")