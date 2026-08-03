import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

# Import all models
from diabetes_model import feature_names as d_features, scaler as d_scaler, best_clf as d_model
from heart_model import feature_names as h_features, scaler as h_scaler, best_clf as h_model
from parkinson_model import feature_names as p_features, scaler as p_scaler, best_clf as p_model

st.set_page_config(page_title="Health Assistant", page_icon="🩺", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
        icons=["activity", "heart", "brain"],
        default_index=0
    )

# ------------------ Diabetes ------------------
if selected == "Diabetes Prediction":
    st.title("🩸 Diabetes Prediction (Logistic Regression)")

    vals = []
    cols = st.columns(3)
    for i, col in enumerate(d_features):
        with cols[i % 3]:
            val = st.number_input(col, min_value=0.0, step=0.1, format="%.2f")
            vals.append(val)

    if st.button("Predict Diabetes"):
        scaled = d_scaler.transform([vals])
        pred = d_model.predict(scaled)[0]
        st.success("🩸 Diabetic" if pred == 1 else "✅ Not Diabetic")

# ------------------ Heart ------------------
elif selected == "Heart Disease Prediction":
    st.title("💓 Heart Disease Prediction (Logistic Regression)")

    vals = []
    cols = st.columns(3)
    for i, col in enumerate(h_features):
        with cols[i % 3]:
            val = st.number_input(col, min_value=0.0, step=0.1, format="%.2f")
            vals.append(val)

    if st.button("Predict Heart Disease"):
        scaled = h_scaler.transform([vals])
        pred = h_model.predict(scaled)[0]
        st.success("💔 Has Heart Disease" if pred == 1 else "❤️ No Heart Disease")

# ------------------ Parkinson's ------------------
else:
    st.title("🧠 Parkinson's Disease Prediction (Logistic Regression)")

    st.markdown(
        "⚙️ *Note:* Some Parkinson’s features can be **negative** or have very small values "
        "(e.g., -0.12345 or 0.00087). Enter values carefully."
    )

    vals = []
    cols = st.columns(3)
    for i, col in enumerate(p_features):
        with cols[i % 3]:
            # Allow negative values and up to 5 decimal places
            val = st.number_input(col, step=0.00001, format="%.5f")
            vals.append(val)

    if st.button("Predict Parkinson's"):
        scaled = p_scaler.transform([vals])
        pred = p_model.predict(scaled)[0]
        st.success("🧠 Has Parkinson's" if pred == 1 else "✅ No Parkinson's")


