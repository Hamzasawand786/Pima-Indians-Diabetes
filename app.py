import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
MODEL_ACCURACY = 0.89   # replace with your real accuracy
PCA_COMPONENTS = pca.n_components_

# -----------------------------
# Custom CSS (Premium UI Look)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f7fbff;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #0b3c5d;
}
.sub-title {
    font-size: 20px;
    color: #3282b8;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.08);
}
.predict-btn button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(90deg,#0f9b8e,#00c6ff);
    color: white;
    border-radius: 10px;
}
.footer {
    text-align: center;
    padding: 15px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown('<div class="main-title">🩺 Diabetes Prediction & Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-based Binary Classification using PCA & Machine Learning</div>', unsafe_allow_html=True)

st.write("""
This system predicts whether a patient is **Diabetic** or **Non-Diabetic** using 
advanced Machine Learning models and PCA-based dimensionality reduction.
It simulates a real-world medical diagnostic dashboard.
""")

st.divider()

# -----------------------------
# Model Information Panel
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🏆 Best Model", BEST_MODEL_NAME)

with col2:
    st.metric("🎯 Model Accuracy", f"{MODEL_ACCURACY*100:.2f}%")

with col3:
    st.metric("📊 PCA Components (95% Variance)", PCA_COMPONENTS)

st.progress(MODEL_ACCURACY)

st.divider()

# -----------------------------
# Patient Input Section
# -----------------------------
st.markdown("### 🧾 Patient Medical Information")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        Pregnancies = st.number_input("🤰 Pregnancies", 0, 20, 1, help="Number of times pregnant")
        Glucose = st.slider("🩸 Glucose Level", 0, 200, 120)
        BloodPressure = st.slider("💓 Blood Pressure", 0, 150, 70)
        SkinThickness = st.slider("📏 Skin Thickness", 0, 100, 20)

    with c2:
        Insulin = st.slider("💉 Insulin Level", 0, 900, 80)
        BMI = st.slider("⚖️ BMI", 0.0, 60.0, 25.0)
        DPF = st.slider("🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        Age = st.slider("🎂 Age", 1, 100, 30)

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict = st.button("🔍 Predict Diabetes Status")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Output Section
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
        st.markdown(
            f"""
            <div class="card" style="border-left:8px solid red;">
            <h2>⚠️ Diabetic (1)</h2>
            <p><b>Confidence:</b> {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div class="card" style="border-left:8px solid green;">
            <h2>✅ Non-Diabetic (0)</h2>
            <p><b>Confidence:</b> {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------
    # Visualization Panel
    # -----------------------------
    st.markdown("## 📊 Visualization Panel")
    v1, v2 = st.columns(2)

    with v1:
        st.markdown("### Prediction Probability")
        fig, ax = plt.subplots()
        ax.bar(["Non-Diabetic", "Diabetic"], probability)
        ax.set_ylim(0,1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

    with v2:
        st.markdown("### Sample Confusion Matrix (Demo)")
        cm = np.array([[90, 10],
                       [15, 85]])
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)

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
