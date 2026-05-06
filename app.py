import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------
data = pickle.load(open("model.pkl", "rb"))
model = data["model"]
model_name = data["model_name"]

# -------------------------------
# TITLE
# -------------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown(f"### 🤖 Model Used: **{model_name}**")

st.warning("⚠️ This is an AI-based prediction tool and not a medical diagnosis.")

# -------------------------------
# SIDEBAR (GUIDE)
# -------------------------------
st.sidebar.header("📘 Input Guide")
st.sidebar.write("""
- Age: Years  
- Cholesterol: <200 normal  
- BP: ~120 normal  
- Exercise Angina: Chest pain during exercise  
""")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("🧾 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 120, 30)

    sex = st.radio("Sex", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0

    cp = st.selectbox("Chest Pain Type", [0,1,2,3])

    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

    chol = st.number_input("Cholesterol", 100, 600, 200)

    fbs = st.radio("Fasting Blood Sugar > 120", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0

with col2:
    restecg = st.selectbox("Resting ECG", [0,1,2])

    thalach = st.slider("Max Heart Rate", 60, 220, 150)

    exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

    slope = st.selectbox("Slope", [0,1,2])

    ca = st.selectbox("Number of Vessels", [0,1,2,3])

    thal = st.selectbox("Thal", [0,1,2,3])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)

    try:
        prob = model.predict_proba(input_data)[0][1]
        prob_text = f"{prob*100:.2f}%"
    except:
        prob = None
        prob_text = "N/A"

    st.subheader("📊 Prediction Result")

    # -------------------------------
    # PROGRESS BAR (COOL FEATURE)
    # -------------------------------
    if prob is not None:
        st.progress(int(prob * 100))

    st.metric(label="Risk Probability", value=prob_text)

    # -------------------------------
    # RISK LEVEL
    # -------------------------------
    if prob is not None:
        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")

    # -------------------------------
    # FINAL RESULT
    # -------------------------------
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # -------------------------------
    # INPUT SUMMARY TABLE
    # -------------------------------
    st.subheader("📋 Input Summary")
    df = pd.DataFrame(input_data, columns=[
        "Age","Sex","CP","BP","Chol","FBS","ECG",
        "HR","Exang","Oldpeak","Slope","CA","Thal"
    ])
    st.dataframe(df)

    # -------------------------------
    # DOWNLOAD REPORT
    # -------------------------------
    report = f"""
Heart Disease Prediction Report

Model Used: {model_name}
Risk Probability: {prob_text}
Result: {"High Risk" if prediction[0]==1 else "Low Risk"}

Note: This is not a medical diagnosis.
"""
    st.download_button("📥 Download Report", report)

# -------------------------------
# MODEL INSIGHTS SECTION (ADVANCED)
# -------------------------------
st.subheader("📈 Model Insights")

st.write("Comparison of models used during training:")

st.bar_chart({
    "Logistic Regression": 0.82,
    "Decision Tree": 0.80,
    "SVM": 0.85
})

# -------------------------------
# FOOTER
# -------------------------------
st.write("---")
st.caption("💡 Developed as part of Data Science Internship | ML-based Healthcare Prediction")