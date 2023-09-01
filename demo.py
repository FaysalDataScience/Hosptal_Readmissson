import streamlit as st
import numpy as np
import pickle

# Load the trained model (assuming you have a Random Forest model saved with this name)
with open('RandomForest_Undersampling_model_10_features.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title('🏥 Predicting Hospital Readmission')

# Collect user input
num_medications = st.number_input('Number of medications:', min_value=0)
num_lab_procedures = st.number_input('Number of lab procedures:', min_value=0)
diag_1 = st.selectbox('Primary diagnosis (diag_1):', ['Yes (250 to 250.99)', 'No'])
time_in_hospital = st.selectbox('Time in hospital:', ['Yes (1 to 14)', 'No'])
discharge_disposition_id = st.selectbox('Discharge disposition ID:', ['Home care', 'Transfer', 'Outpatients', 'Expired Home/Medical', 'Undefined'])
admission_source_id = st.selectbox('Admission source ID:', ['Referral', 'Transfer', 'Undefined', 'Newborn'])
number_inpatient_log1p = st.number_input('Number of inpatient events:', min_value=0)
number_diagnoses = st.number_input('Number of diagnoses:', min_value=0)
age = st.number_input('Age:', min_value=0)
num_procedures = st.number_input('Number of procedures:', min_value=0)

# Prepare data
user_data = np.array([
    num_medications,
    num_lab_procedures,
    1 if diag_1 == 'Yes (250 to 250.99)' else 0,
    1 if time_in_hospital == 'Yes (1 to 14)' else 0,
    number_inpatient_log1p,
    number_diagnoses,
    age,
    num_procedures,
    1 if discharge_disposition_id == 'Home care' else 2 if discharge_disposition_id == 'Transfer' else 10 if discharge_disposition_id == 'Outpatients' else 11 if discharge_disposition_id == 'Expired Home/Medical' else 18,
    1 if admission_source_id == 'Referral' else 4 if admission_source_id == 'Transfer' else 9 if admission_source_id == 'Undefined' else 11
]).reshape(1, -1)

# Add a button for predictions
if st.button("Submit"):
    # Make prediction
    prediction_proba = model.predict_proba(user_data)
    prob_of_readmission = prediction_proba[0][1]  # Probability of being readmitted
    
    # Show prediction
    st.markdown(f'<span style="color:red">📊 The model predicts a {prob_of_readmission * 100:.2f}% probability of being readmitted.</span>', unsafe_allow_html=True)