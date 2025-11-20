import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app
st.title("🎓 Student Performance Prediction")

st.write("Enter the student's details to predict the final score:")

# Input fields (matching your CSV columns)
study_hours = st.number_input("Study Hours", min_value=0, max_value=10, value=2)
attendance_percent = st.number_input("Attendance Percentage", min_value=0, max_value=100, value=60)
prev_grade = st.number_input("Previous Grade", min_value=0, max_value=100, value=55)
family_support = st.selectbox("Family Support", ['yes', 'no'])
extracurricular = st.selectbox("Extracurricular Activities", ['yes', 'no'])

# Convert yes/no to numeric (same as model)
family_support_num = 1 if family_support == 'yes' else 0
extracurricular_num = 1 if extracurricular == 'yes' else 0

# Predict button
if st.button("Predict Final Grade"):
    # Create input array
    input_data = np.array([[study_hours, attendance_percent, prev_grade, family_support_num, extracurricular_num]])
    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Final Score: {prediction:.2f}")