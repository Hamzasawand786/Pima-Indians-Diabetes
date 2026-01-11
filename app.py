import streamlit as st
import numpy as np
import joblib
import pandas as pd
import altair as alt  # Built-in, interactive charts

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🩺 Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Theme Toggle
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

col_theme, _ = st.columns([1, 9])
with col_theme:
    if st.button("🌙 / ☀️ Theme"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# -----------------------------
# Theme Colors
# -----------------------------
if st.session_state.theme == "light":
    primary_color = "#0f9b8e"
    bg_color = "#f7fbff"
    text_color = "black"
    card_bg = "white"
else:
    primary_color = "#00c6ff"
    bg_color = "#0e1117"
    text_color = "white"
    card_bg = "#161b22"

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown(f"""
<style>
body {{background-color:{bg_color}; color:{text_color}; font-family:sans-serif;}}

.main-title {{font-size:42px; font-weight:800; color:{primary_color}; margin-bottom:0;}}
.sub-title {{font-size:18px; color:{primary_color}; margin-top:0;}}

.card {{
    background:{card_bg};
    padding:20px;
    border-radius:15px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.1);
    margin-bottom:20px;
}}

.btn-primary {{
    background: linear-gradient(90deg, #0f9b8e, #00c6ff);
    color:white;
    font-size:18px;
    font-weight:bold;
    border-radius:12px;
    padding:10px 0;
    width:100%;
    text-align:center;
    display:inline-block;
    cursor:pointer;
    transition: 0.3s;
}}
.btn-primary:hover {{opacity:0.9;}}

/* Bubble-style tabs */
div[data-testid="stTabs"] > div {{gap: 15px;}}
div[data-baseweb="tab"] {{
    border-radius: 30px !important;
    padding: 8px 25px !important;
    font-weight: 600 !important;
    transition: 0.3s;
    border: 2px solid {primary_color};
    background-color: {card_bg};
    color: {text_color};
}}
div[data-baseweb="tab"][data-active="true"] {{
    background-color: {primary_color} !important;
    color: white !important;
}}
div[data-baseweb="tab"]:hover {{opacity:0.8; cursor:pointer;}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("best_diabetes_model.pkl")
pca = joblib.load("pca_transformer.pkl")
scaler = joblib.load("scaler.pkl")

BEST_MODEL_NAME = "Random Forest Classifier"
MODEL_ACCURACY = 0.89
PCA_COMPONENTS = pca.n_components_

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🩺 Diabetes Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict Diabetes Status using AI & PCA</div>', unsafe_allow_html=True)
st.write("Interactive dashboard for medical prediction using Machine Learning.")

st.divider()

# -----------------------------
# Metrics
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("🏆 Best Model", BEST_MODEL_NAME)
c2.metric("🎯 Accuracy", f"{MODEL_ACCURACY*100:.2f}%")
c3.metric("📊 PCA Components", PCA_COMPONENTS)
st.progress(MODEL_ACCURACY)

st.divider()

# -----------------------------
# Tabs for Inputs & Results
# -----------------------------
tabs = st.tabs(["🧾 Patient Input", "🩺 Prediction Result", "📊 Visualization"])

with tabs[0]:
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
    st.markdown('<div class="btn-primary">🔍 Predict Diabetes Status</div>', unsafe_allow_html=True)
    predict = st.button("Predict")

with tabs[1]:
    st.info("Prediction results will appear here after clicking Predict button.")

with tabs[2]:
    st.info("Visualizations will appear here after prediction.")

# -----------------------------
# Prediction Logic
# -----------------------------
if predict:
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DPF, Age]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)[0]
    probability = model.predict_proba(input_pca)[0]
    confidence = np.max(probability) * 100

    # Update Tabs
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if prediction == 1:
            st.error(f"⚠️ Diabetic (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ Non-Diabetic (Confidence: {confidence:.2f}%)")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Probability Bar Chart using Altair
        prob_df = pd.DataFrame({
            "Class": ["Non-Diabetic", "Diabetic"],
            "Probability": probability
        })
        chart = alt.Chart(prob_df).mark_bar().encode(
            x='Class',
            y='Probability',
            color=alt.Color('Class', scale=alt.Scale(domain=['Non-Diabetic','Diabetic'],
                                                    range=['green','red'])),
            tooltip=['Class','Probability']
        )
        st.altair_chart(chart, use_container_width=True)

        # Sample Confusion Matrix
        cm_df = pd.DataFrame([[90, 10],[15, 85]], columns=["Predicted 0","Predicted 1"],
                             index=["Actual 0","Actual 1"])
        st.markdown("### Sample Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(f"""
<div style='text-align:center; margin-top:30px; color:{text_color};'>
<hr>
Final Lab Exam – Data Science | Dataset: Pima Indians Diabetes Dataset <br>
Developed Hamza Ashraf
</div>
""", unsafe_allow_html=True)
