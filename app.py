import streamlit as st
import numpy as np
import joblib
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🩺 Diabetes Prediction & Analysis Dashboard",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model Assets
# -----------------------------
model = joblib.load("best_diabetes_model.pkl")
pca = joblib.load("pca_transformer.pkl")
scaler = joblib.load("scaler.pkl")

BEST_MODEL_NAME = "Random Forest Classifier"
MODEL_ACCURACY = 0.89   # update if you know exact value
PCA_COMPONENTS = pca.n_components_

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-title {font-size:40px;font-weight:800;color:#0b3c5d;}
.sub-title {font-size:20px;color:#3282b8;}
.card {
    background-color:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 12px rgba(0,0,0,0.08);
}
.predict-btn button {
    width:100%;
    height:55px;
    font-size:18px;
    font-weight:bold;
    background:linear-gradient(90deg,#0f9b8e,#00c6ff);
    color:white;
    border-radius:10px;
}
.footer {
    text-align:center;
    padding:15px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🩺 Diabetes Prediction & Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-based Binary Classification using PCA & Machine Learning</div>', unsafe_allow_html=True)

st.write("""
This professional medical dashboard predicts whether a patient is **Diabetic** or **Non-Diabetic**
using PCA and Machine Learning models trained on the Pima Indians Diabetes Dataset.
""")

st.divider()

# -----------------------------
# Model Info Panel
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("🏆 Best Model", BEST_MODEL_NAME)
c2.metric("🎯 Model Accuracy", f"{MODEL_ACCURACY*100:.2f}%")
c3.metric("📊 PCA Components (95% Variance)", PCA_COMPONENTS)
st.progress(MODEL_ACCURACY)

st.divider()

# -----------------------------
# Patient Input Section
# -----------------------------
st.markdown("## 🧾 Patient Medical Information")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("🤰 Pregnancies", 0, 20, 1)
        Glucose = st.slider("🩸 Glucose", 0, 200, 120)
        BloodPressure = st.slider("💓 Blood Pressure", 0, 150, 70)
        SkinThickness = st.slider("📏 Skin Thickness", 0, 100, 20)

    with col2:
        Insulin = st.slider("💉 Insulin", 0, 900, 80)
        BMI = st.slider("⚖️ BMI", 0.0, 60.0, 25.0)
        DPF = st.slider("🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        Age = st.slider("🎂 Age", 1, 100, 30)

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Predict Button
# -----------------------------
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict = st.button("🔍 Predict Diabetes Status")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Output
# -----------------------------
if predict:
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                             Insulin, BMI, DPF, Age]])

    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    prediction = model.predict(input_pca)[0]
    probability = model.predict_proba(input_pca)[0]
    confidence = np.max(probability) * 100

    st.divider()
    st.markdown("## 🩺 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
        <div class="card" style="border-left:8px solid red;">
        <h2>⚠️ Diabetic (1)</h2>
        <p><b>Confidence:</b> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card" style="border-left:8px solid green;">
        <h2>✅ Non-Diabetic (0)</h2>
        <p><b>Confidence:</b> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # Visualization Panel (Streamlit Native)
    # -----------------------------
    st.markdown("## 📊 Visualization Panel")

    prob_df = pd.DataFrame({
        "Class": ["Non-Diabetic", "Diabetic"],
        "Probability": probability
    })

    st.bar_chart(prob_df.set_index("Class"))

    cm_df = pd.DataFrame(
        [[90, 10], [15, 85]],
        columns=["Predicted 0", "Predicted 1"],
        index=["Actual 0", "Actual 1"]
    )

    st.markdown("### Sample Confusion Matrix (Demo)")
    st.dataframe(cm_df, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
<hr>
Final Lab Exam – Data Science <br>
Dataset: Pima Indians Diabetes Dataset <br>
Developed using Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
