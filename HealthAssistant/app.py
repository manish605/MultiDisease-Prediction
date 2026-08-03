import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import pickle
from pathlib import Path

# Determine base directory (the directory containing this app.py)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Health Assistant", page_icon="🩺", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
        icons=["activity", "heart", "brain"],
        default_index=0
    )

# Helper to load a pickled model bundle (dict with keys: model, scaler, features)
def load_model_bundle(filename):
    path = MODELS_DIR / filename
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except FileNotFoundError:
        st.error(f"Model file not found: {path}.\nPlease run the training script `HealthAssistant/train_models.py` locally to generate model files and commit them, or upload the .sav files to {MODELS_DIR}.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model file {path}: {e}")
        st.stop()

# Load model bundles (each bundle is a dict: { 'model': ..., 'scaler': ..., 'features': [...] })
d_bundle = load_model_bundle("diabetes_model.sav")
h_bundle = load_model_bundle("heart_model.sav")
p_bundle = load_model_bundle("parkinsons_model.sav")

# Unpack
_d_model = d_bundle["model"]
d_scaler = d_bundle["scaler"]
d_features = d_bundle["features"]

_h_model = h_bundle["model"]
h_scaler = h_bundle["scaler"]
h_features = h_bundle["features"]

_p_model = p_bundle["model"]
p_scaler = p_bundle["scaler"]
p_features = p_bundle["features"]

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
        pred = _d_model.predict(scaled)[0]
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
        pred = _h_model.predict(scaled)[0]
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
        pred = _p_model.predict(scaled)[0]
        st.success("🧠 Has Parkinson's" if pred == 1 else "✅ No Parkinson's")
