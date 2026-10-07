import streamlit as st
import joblib
import pandas as pd
model = joblib.load("model.pkl")
st.title("Student Stress Level Predictor")

st.write("Enter your daily lifestyle details to predict your stress level.")
study_hours = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0)
sleep_hours = st.number_input("Sleep Hours Per Day", min_value=0.0, max_value=24.0)
extracurricular_hours = st.number_input("Extracurricular Hours Per Day", min_value=0.0, max_value=24.0)
social_hours = st.number_input("Social Hours Per Day", min_value=0.0, max_value=24.0)
physical_activity_hours = st.number_input("Physical Activity Hours Per Day", min_value=0.0, max_value=24.0)
gpa = st.number_input("GPA", min_value=0.0, max_value=4.0)
input_data = {
    "Study_Hours_Per_Day": study_hours,
    "Extracurricular_Hours_Per_Day": extracurricular_hours,
    "Sleep_Hours_Per_Day": sleep_hours,
    "Social_Hours_Per_Day": social_hours,
    "Physical_Activity_Hours_Per_Day": physical_activity_hours,
    "GPA": gpa
}
input_df = pd.DataFrame([input_data])
if st.button("Predict Stress Level"):
    prediction = model.predict(input_df)
    stress_level = prediction[0]
    probabilities = model.predict_proba(input_df)
    confidence = probabilities.max() * 100

    st.markdown("### 💡 Lifestyle Insights")

    if study_hours > 8:
        st.warning("📚 Your study hours are relatively high.")

    if sleep_hours < 6:
        st.info("😴 Your sleep duration is relatively low.")

    if physical_activity_hours < 2:
        st.info("🏃 Your physical activity hours are relatively low.")

    st.markdown("### 🎯 Your Stress Level")

    st.markdown(
        f"""
        <div style="
            padding: 25px;
            border-radius: 15px;
            border: 2px solid #888;
            text-align: center;
            margin: 20px 0;
        ">
            <h2>{stress_level}</h2>
            <p>Predicted based on your lifestyle and academic inputs.</p>
            <p>Model Confidence: {confidence:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )