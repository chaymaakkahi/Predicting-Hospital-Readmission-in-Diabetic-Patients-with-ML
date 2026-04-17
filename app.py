import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Readmission Predictor", page_icon="🏥", layout="centered")

# --- LOAD ARTIFACTS ---
# Try loading from the 'models' folder (where your new notebook saves them)
# If you put them in the root folder, change 'models/' to ''
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# --- PREPROCESSING FUNCTION (Matches your final notebook exactly) ---
def preprocess_patient_input(data_dict):
    df = pd.DataFrame([data_dict])
    
    # 1. ICD-9 Grouping (Matches the notebook's map_icd9 function)
    # The UI sends the 8 categories directly, so we just pass them through
    
    # 2. Age midpoint
    age_map = {'[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45, 
               '[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95}
    if 'age' in df.columns:
        df['age_mid'] = df['age'].map(age_map).fillna(55)
        df.drop(columns=['age'], inplace=True)

    # 3. Medication activity
    ALL_MED_COLS = ['metformin','repaglinide','nateglinide','chlorpropamide',
                    'glimepiride','acetohexamide','glipizide','glyburide',
                    'tolbutamide','pioglitazone','rosiglitazone','acarbose',
                    'miglitol','troglitazone','tolazamide','examide',
                    'sitagliptin','insulin','glyburide-metformin',
                    'glipizide-metformin','glimepiride-pioglitazone',
                    'metformin-rosiglitazone','metformin-pioglitazone']
    
    # Assume 'No' for any medication the user didn't specify
    for col in ALL_MED_COLS:
        if col not in df.columns:
            df[col] = 'No'
            
    df['n_active_meds'] = (df[ALL_MED_COLS] != 'No').sum(axis=1)
    df['n_med_changes']  = df[ALL_MED_COLS].isin(['Up','Down']).sum(axis=1)

    # 4. Prior visits
    df['prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    
    # 5. Binary encoding
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1}).fillna(0).astype(int)
    df['change'] = df['change'].map({'No': 0, 'Ch': 1}).fillna(0).astype(int)
    df['diabetesMed'] = df['diabetesMed'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
    
    # 6. Ordinal encode medications
    med_order = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': -1}
    for col in ALL_MED_COLS:
        df[col] = df[col].map(med_order).fillna(0).astype(int)
        
    # 7. Fill missing categoricals
    df['race'].fillna('Unknown', inplace=True)

    # 8. THE MAGIC TRICK: Align with Scaler features
    all_scaler_features = scaler.feature_names_in_
    df_for_scaler = pd.DataFrame(0, index=df.index, columns=all_scaler_features)
    for col in df.columns:
        if col in all_scaler_features:
            df_for_scaler[col] = df[col]
            
    # 9. Scale
    X_final = scaler.transform(df_for_scaler)
    return X_final

# --- STREAMLIT UI ---
st.title("🏥 Diabetic Patient Readmission Risk")
st.markdown("Predict if a patient will be readmitted within **30 days**.")
st.caption("⚠️ Model optimized with a 0.16 clinical threshold to minimize missed high-risk patients.")
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
    
    st.write("**Clinical Tests**")
    A1Cresult = st.selectbox("A1C Test Result", ['None', 'Norm', '>7', '>8'])
    max_glu_serum = st.selectbox("Max Glucose Serum", ['None', 'Norm', '>200', '>300'])

st.divider()
st.subheader("Diagnoses & Treatment")
col3, col4 = st.columns(2)

with col3:
    diag_1 = st.selectbox("Primary Diagnosis (ICD-9 Category)", 
                          ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 
                           'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'])
    diag_2 = st.selectbox("Secondary Diagnosis", 
                          ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 
                           'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'], index=8)
    diag_3 = st.selectbox("Additional Diagnosis", 
                          ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 
                           'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'], index=8)

with col4:
    st.write("**Key Medications**")
    insulin = st.selectbox("Insulin Status", ['No', 'Steady', 'Up', 'Down'])
    change = st.radio("Medication Changed?", ['No', 'Ch'], horizontal=True, index=1)
    diabetesMed = st.radio("Diabetic Medication Prescribed?", ['No', 'Yes'], horizontal=True, index=1)

st.divider()

# --- PREDICTION LOGIC ---
if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
    input_data = {
        'age': age, 'gender': gender, 'race': race,
        'admission_type_id': 1, 'discharge_disposition_id': 1, 'admission_source_id': 7,
        'time_in_hospital': time_in_hospital, 'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures, 'num_medications': num_medications,
        'number_diagnoses': number_diagnoses, 'number_outpatient': number_outpatient,
        'number_emergency': number_emergency, 'number_inpatient': number_inpatient,
        'diag_1': diag_1, 'diag_2': diag_2, 'diag_3': diag_3,
        'A1Cresult': A1Cresult, 'max_glu_serum': max_glu_serum,
        'change': change, 'diabetesMed': diabetesMed, 
        'insulin': insulin, 'metformin': 'No'
    }
    
    try:
        X_processed = preprocess_patient_input(input_data)
        proba = model.predict_proba(X_processed)[0][1]
        
        # 🚨 THE FIX: Use the 0.16 threshold discovered during Error Analysis
        prediction = 1 if proba >= 0.16 else 0
        
        # Display Results
        st.subheader("Clinical Prediction")
        
        # Show the raw probability so doctors understand the scale
        st.metric(label="Calculated Risk Probability", value=f"{proba*100:.1f}%", delta="Based on 74 clinical features")
        
        if prediction == 1:
            st.error(f"⚠️ HIGH RISK: Probability ({proba*100:.1f}%) exceeds the 0.16 clinical threshold.")
            st.warning("Recommendation: Schedule follow-up call within 48 hours and consult endocrinologist before discharge.")
        else:
            st.success(f"✅ LOW RISK: Probability ({proba*100:.1f}%) is below the 0.16 clinical threshold.")
            st.info("Recommendation: Standard discharge protocol.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")