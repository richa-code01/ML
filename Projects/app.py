import streamlit as st
import pandas as pd
import joblib

# Load the trained model, features and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Set up the Streamlit app
st.title('Heart Disease Prediction App')
st.write('Enter the following details to predict the likelihood of heart disease:')

age = st.slider('Age', 20, 100, 50)
sex = st.selectbox("Sex", ['Male', 'Female'])
pain = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
restbp = st.slider('Resting Blood Pressure', 80, 200, 120)
chol = st.slider('Serum Cholesterol', 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['True', 'False'])
restecg = st.selectbox("Resting ECG", ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ['Upsloping', 'Flat', 'Downsloping'])
ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 0)
thal = st.selectbox("Thalassemia", ['Normal', 'Fixed Defect', 'Reversible Defect'])
chest_pain_mapping = {
    'Typical Angina': 0,
    'Atypical Angina': 1,
    'Non-Anginal Pain': 2,
    'Asymptomatic': 3
}
restecg_mapping = {
    'Normal': 0,
    'ST-T Wave Abnormality': 1,
    'Left Ventricular Hypertrophy': 2
}
exang_mapping = {
    'Yes': 1,
    'No': 0
}
slope_mapping = {
    'Upsloping': 0,
    'Flat': 1,
    'Downsloping': 2
}
thal_mapping = {
    'Normal': 1,
    'Fixed Defect': 2,
    'Reversible Defect': 3
}
input_data = {
    'age': age,
    'sex': 1 if sex == 'Male' else 0,
    'cp': chest_pain_mapping[pain],
    'trestbps': restbp,
    'chol': chol,
    'fbs': 1 if fbs == 'True' else 0,
    'restecg': restecg_mapping[restecg],
    'thalach': thalach,
    'exang': exang_mapping[exang],
    'oldpeak': oldpeak,
    'slope': slope_mapping[slope],
    'ca': ca,
    'thal': thal_mapping[thal]
}

if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df[feature_names])
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error('The model predicts that you have a high likelihood of heart disease. Please consult a doctor for further evaluation.')
    else:
        st.success('The model predicts that you have a low likelihood of heart disease. Keep maintaining a healthy lifestyle!')