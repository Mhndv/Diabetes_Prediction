import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load trained model and scaler
model = tf.keras.models.load_model("model_diabetes_2.h5")
scaler = joblib.load("scalar_diabeter.pkl")

# Set page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# App title and instructions
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient medical data to predict diabetes risk.")

# Input form for features
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=1)

    submitted = st.form_submit_button("Predict")

# Function to preprocess input
def preprocess_input(values):
    input_array = np.array(values).reshape(1, -1)
    return scaler.transform(input_array)

# When form is submitted
if submitted:
    user_input = [
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]
    
    processed_input = preprocess_input(user_input)
    pred_prob = model.predict(processed_input)[0][0]
    pred_label = "🟢 No Diabetes" if pred_prob < 0.5 else "🔴 Diabetes"

    st.subheader("🔍 Prediction Result")
    st.write(f"**Result:** {pred_label}")
    st.write(f"**Probability:** {pred_prob:.2%}")