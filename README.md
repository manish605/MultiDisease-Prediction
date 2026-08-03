# MultiDisease-Prediction (Health Assistant)

Project overview
----------------
A Streamlit app that predicts whether a person has Diabetes, Heart Disease, or Parkinson's from user-provided features. The app loads pre-trained, serialized model bundles for fast predictions and is prepared for deployment on Streamlit Community Cloud.

Features
--------
- Predict Diabetes, Heart Disease, and Parkinson's via an easy-to-use Streamlit UI
- Fast startup by loading pre-trained models (no retraining on app launch)
- One-time training script included to regenerate model bundles if needed
- Clear error messages if model files are missing

Technologies Used
-----------------
- Python 3.x
- Streamlit (UI)
- scikit-learn (models)
- pandas, numpy (data handling)
- matplotlib, seaborn (local evaluation)
- streamlit-option-menu

Installation
------------
1. Clone the repo:
   git clone https://github.com/manish605/MultiDisease-Prediction.git
2. (Optional) Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate   # Windows
3. Install dependencies:
   pip install -r requirements.txt

One-time local training (produces model bundles)
------------------------------------------------
The Streamlit app expects serialized model bundles in `HealthAssistant/models/`. To generate them locally:

cd HealthAssistant
python train_models.py

This creates:
- HealthAssistant/models/diabetes_model.sav
- HealthAssistant/models/heart_model.sav
- HealthAssistant/models/parkinsons_model.sav

Commit those files to the repo so Streamlit Community Cloud can load them.

How to run the Streamlit app
----------------------------
From repository root:
streamlit run HealthAssistant/app.py

Deployment on Streamlit Community Cloud
--------------------------------------
1. Ensure `HealthAssistant/models/*.sav` are committed to the repository (or provide a hosted download and adjust the app).
2. Add the repo to Streamlit Community Cloud and set the app entrypoint to:
   HealthAssistant/app.py
3. Streamlit Cloud will install packages from `requirements.txt` and run the app.

Project structure
-----------------
- HealthAssistant/
  - app.py                # Streamlit app (loads serialized models)
  - train_models.py       # Script to train and save model bundles
  - models/               # Serialized model bundles (.sav) — should be committed
  - diabetes_model.py     # legacy training file (not used by app)
  - heart_model.py        # legacy training file (not used by app)
  - parkinson_model.py    # legacy training file (not used by app)
- datasets/               # CSV files required for training
- README.md
- requirements.txt
- .gitignore

Notes
-----
- Each model bundle is a pickle of a dict: {"model": estimator, "scaler": scaler, "features": [names]}.
- If model files are large (>50 MB) consider Git LFS or hosting them externally.
- Use the included `train_models.py` script to regenerate model bundles locally if you need to update models.
