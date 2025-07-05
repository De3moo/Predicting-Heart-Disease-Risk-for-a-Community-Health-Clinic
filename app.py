import streamlit as st
import pickle
import numpy as np
import os
import joblib

# Load model components
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
model = joblib.load(os.path.join(MODEL_DIR, 'heart_disease_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

# Categorical feature options
CATEGORICAL_OPTIONS = {
    'HeartDisease': ['no', 'yes'],
    'Smoking': ['yes', 'no'],
    'AlcoholDrinking': ['no', 'yes'],
    'Stroke': ['no', 'yes'],
    'DiffWalking': ['no', 'yes'],
    'Sex': ['female', 'male'],
    'AgeCategory': ['55-59', '80 or older', '65-69', '75-79', '40-44', '70-74', '60-64', '50-54', '45-49', '18-24', '35-39', '30-34', '25-29'],
    'Race': ['white', 'black', 'asian', 'american indian/alaskan native', 'other', 'hispanic'],
    'Diabetic': ['yes', 'no', 'no, borderline diabetes', 'yes (during pregnancy)'],
    'PhysicalActivity': ['yes', 'no'],
    'GenHealth': ['very good', 'fair', 'good', 'poor', 'excellent'],
    'Asthma': ['yes', 'no'],
    'KidneyDisease': ['no', 'yes'],
    'SkinCancer': ['yes', 'no'],
}

# Features
FEATURES = feature_columns

# Helper: Predict risk
def predict_risk(input_data_dict):
    # Prepare input in feature order
    X = []
    for feat in FEATURES:
        val = input_data_dict[feat]
        # Normalize string inputs
        if isinstance(val, str):
            val = val.strip().lower()
        # Encode categorical
        if feat in label_encoders:
            le = label_encoders[feat]
            # Handle unseen categories
            if val not in le.classes_:
                # Add unseen value to classes_
                le.classes_ = np.append(le.classes_, val)
            val = le.transform([val])[0]
        X.append(val)
    X = np.array(X).reshape(1, -1)
    # Scale numeric features
    # Find indices of numeric features
    numeric_indices = [i for i, feat in enumerate(FEATURES) if feat not in label_encoders]
    if numeric_indices:
        X_numeric = X[:, numeric_indices]
        X[:, numeric_indices] = scaler.transform(X_numeric)
    # Predict
    proba = model.predict_proba(X)[0]
    pred = np.argmax(proba)
    confidence = proba[pred]
    label = 'At Risk' if pred == 1 else 'Not at Risk'
    return label, confidence

# Streamlit UI
st.title('Heart Disease Risk Predictor')
st.write('Fill in the form below to predict your risk of heart disease.')

with st.form('risk_form'):
    user_input = {}
    for feat in FEATURES:
        if feat in CATEGORICAL_OPTIONS:
            options = CATEGORICAL_OPTIONS[feat]
            default = options[0]
            user_input[feat] = st.selectbox(feat, options, index=0)
        elif feat in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']:
            # Numeric features
            min_val, max_val = (0, 100) if feat != 'SleepTime' else (0, 24)
            user_input[feat] = st.number_input(feat, min_value=float(min_val), max_value=float(max_val), value=0.0)
        else:
            user_input[feat] = st.text_input(feat, '')
    submitted = st.form_submit_button('Predict Risk')

if submitted:
    # Lowercase and strip all string inputs
    for k, v in user_input.items():
        if isinstance(v, str):
            user_input[k] = v.strip().lower()
    label, confidence = predict_risk(user_input)
    st.subheader('Prediction:')
    st.write(f'**{label}**')
    st.write(f'Confidence: {confidence*100:.2f}%') 