import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Readmission Predictor", page_icon="🏥", layout="centered")

# --- LOAD ARTIFACTS (Only the 3 files you have) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('selected_features.pkl')
    return model, scaler, features

model, scaler, selected_features = load_artifacts()

# --- PREPROCESSING FUNCTION ---
def preprocess_patient_input(data_dict):
    df = pd.DataFrame([data_dict])
    
    # 1. Feature Engineering
    age_map = {'[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45, 
               '[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95}
    df['age_mid'] = df['age'].map(age_map).fillna(55)
    
    med_order = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': -1}
    all_possible_meds = ['metformin', 'insulin', 'glipizide', 'glyburide']
    med_cols_present = [c for c in all_possible_meds if c in df.columns]
    
    for col in med_cols_present:
        df[col] = df[col].map(med_order).fillna(0).astype(int)
        
    if len(med_cols_present) > 0:
        df['n_active_meds'] = (df[med_cols_present] != 0).sum(axis=1)
        df['n_med_changes'] = df[med_cols_present].isin([2, -1]).sum(axis=1)
    else:
        df['n_active_meds'] = 0
        df['n_med_changes'] = 0
        
    # 2. Categorical Encoding
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1}).fillna(0)
    df['change'] = df['change'].map({'No': 0, 'Ch': 1}).fillna(0)
    df['diabetesMed'] = df['diabetesMed'].map({'No': 0, 'Yes': 1}).fillna(0)
    
    # One-Hot Encoding
    df = pd.get_dummies(df, drop_first=True)
    
    # 3. GET ALL FEATURES THE SCALER EXPECTS (This is the magic trick!)
    # The scaler remembers every single column from the notebook training
    all_scaler_features = scaler.feature_names_in_
    
    # Create a dataframe with ALL original columns, filling missing user inputs with 0
    df_for_scaler = pd.DataFrame(0, index=df.index, columns=all_scaler_features)
    for col in df.columns:
        if col in all_scaler_features:
            df_for_scaler[col] = df[col]
            
    # 4. SCALE FIRST (Now the scaler is happy because it sees all columns)
    X_scaled = scaler.transform(df_for_scaler)
    
    # 5. FEATURE SELECTION SECOND (Pick only the columns the model needs)
    # Find the exact position of our selected features in the scaled array
    feature_indices = [list(all_scaler_features).index(f) for f in selected_features]
    X_final = X_scaled[:, feature_indices]
    
    return X_final

# --- STREAMLIT UI ---
st.title("🏥 Diabetic Patient Readmission Risk")
st.markdown("Predict if a patient will be readmitted within **30 days** based on clinical data.")

st.divider()
st.subheader("Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    st.write("**Demographics & History**")
    age = st.selectbox("Age Bracket", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                                        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=6)
    gender = st.radio("Gender", ['Female', 'Male'], horizontal=True)
    race = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'])
    
    number_outpatient = st.number_input("Outpatient Visits (Past Year)", 0, 50, 0)
    number_emergency = st.number_input("Emergency Visits (Past Year)", 0, 50, 0)
    number_inpatient = st.number_input("Inpatient Visits (Past Year)", 0, 50, 0)

with col2:
    st.write("**Current Hospital Stay**")
    time_in_hospital = st.number_input("Days in Hospital", 1, 14, 4)
    num_lab_procedures = st.number_input("Number of Lab Procedures", 1, 130, 40)
    num_procedures = st.number_input("Number of Procedures", 0, 10, 1)
    num_medications = st.number_input("Number of Medications", 1, 80, 15)
    number_diagnoses = st.number_input("Number of Diagnoses", 1, 16, 8)
    
    st.write("**Treatment Details**")
    change = st.radio("Medication Changed?", ['No', 'Ch'], horizontal=True, index=1)
    diabetesMed = st.radio("Diabetic Medication Prescribed?", ['No', 'Yes'], horizontal=True, index=1)
    insulin = st.selectbox("Insulin Status", ['No', 'Steady', 'Up', 'Down'])
    metformin = st.selectbox("Metformin Status", ['No', 'Steady', 'Up', 'Down'])

st.divider()

# --- PREDICTION LOGIC ---
if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
    input_data = {
        'age': age, 'gender': gender, 'race': race,
        'time_in_hospital': time_in_hospital, 'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures, 'num_medications': num_medications,
        'number_diagnoses': number_diagnoses, 'number_outpatient': number_outpatient,
        'number_emergency': number_emergency, 'number_inpatient': number_inpatient,
        'change': change, 'diabetesMed': diabetesMed, 'insulin': insulin, 'metformin': metformin
    }
    
    try:
        X_processed = preprocess_patient_input(input_data)
        
        # Predict Probability
        proba = model.predict_proba(X_processed)[0][1]
        
        # Use standard 0.50 threshold since we don't have the custom one
        prediction = 1 if proba >= 0.50 else 0
        
        # Display Results
        st.subheader("Clinical Prediction")
        
        if prediction == 1:
            st.error(f"⚠️ HIGH RISK: {proba*100:.1f}% probability of readmission < 30 days.")
            st.warning("Recommendation: Schedule follow-up call within 48 hours and consult endocrinologist before discharge.")
        else:
            st.success(f"✅ LOW RISK: {proba*100:.1f}% probability of readmission < 30 days.")
            st.info("Recommendation: Standard discharge protocol.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")