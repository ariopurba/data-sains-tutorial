import streamlit as st
import joblib

# Load model
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Diagnosis Diabetes')

# Input form
pregnancies = st.number_input('Kehamilan')
glucose = st.number_input('Kadar Glukosa')
blood_pressure = st.number_input('Tekanan Darah')
skin_thickness = st.number_input('Ketebalan Kulit')
insulin = st.number_input('Insulin')
bmi = st.number_input('BMI')
dpf = st.number_input('Riwayat Keluarga Diabetes')
age = st.number_input('Usia')

if st.button('Prediksi'):
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success("Hasil: Diabetes" if prediction[0] == 1 else "Hasil: Sehat")