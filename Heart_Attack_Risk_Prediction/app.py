import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="🫀", layout="centered")

# -------------------------------------------------------
# Load the already-trained model and encoder (saved from
# the notebook) instead of retraining inside the app
# -------------------------------------------------------
@st.cache_resource
def load_model():
    with open("dt_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le


model, le = load_model()

# Column order must exactly match the order used during training in the notebook
feature_order = [
    "Age", "Sex", "Cholesterol", "Heart Rate", "Diabetes", "Family History",
    "Smoking", "Obesity", "Alcohol Consumption", "Exercise Hours Per Week",
    "Diet", "Previous Heart Problems", "Medication Use", "Stress Level",
    "Sedentary Hours Per Day", "Income", "BMI", "Triglycerides",
    "Physical Activity Days Per Week", "Sleep Hours Per Day",
    "Systolic_BP", "Diastolic_BP"
]

# -------------------------------------------------------
# Page layout
# -------------------------------------------------------
st.title("🫀 Heart Attack Risk Predictor")
st.write("Enter patient details below to estimate heart attack risk using a Decision Tree model.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 90, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cholesterol = st.slider("Cholesterol", 120, 400, 220)
    heart_rate = st.slider("Heart Rate", 40, 110, 75)
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    obesity = st.selectbox("Obesity", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    exercise_hrs = st.slider("Exercise Hours Per Week", 0.0, 20.0, 5.0)
    diet = st.selectbox("Diet", ["Unhealthy", "Average", "Healthy"])

with col2:
    prev_heart_problems = st.selectbox("Previous Heart Problems", ["No", "Yes"])
    medication_use = st.selectbox("Medication Use", ["No", "Yes"])
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    sedentary_hrs = st.slider("Sedentary Hours Per Day", 0.0, 12.0, 6.0)
    income = st.number_input("Income", 20000, 300000, 60000, step=1000)
    bmi = st.slider("BMI", 18.0, 40.0, 25.0)
    triglycerides = st.slider("Triglycerides", 30, 800, 250)
    activity_days = st.slider("Physical Activity Days Per Week", 0, 7, 3)
    sleep_hrs = st.slider("Sleep Hours Per Day", 4, 10, 7)
    systolic_bp = st.slider("Systolic Blood Pressure", 90, 180, 120)
    diastolic_bp = st.slider("Diastolic Blood Pressure", 60, 110, 80)

st.divider()

# -------------------------------------------------------
# Build input row in the SAME order/encoding as training
# -------------------------------------------------------
yes_no = {"No": 0, "Yes": 1}
diet_map = {"Unhealthy": 0, "Average": 1, "Healthy": 2}

input_dict = {
    "Age": age,
    "Sex": le.transform([sex])[0],   # uses the SAME LabelEncoder saved from the notebook
    "Cholesterol": cholesterol,
    "Heart Rate": heart_rate,
    "Diabetes": yes_no[diabetes],
    "Family History": yes_no[family_history],
    "Smoking": yes_no[smoking],
    "Obesity": yes_no[obesity],
    "Alcohol Consumption": yes_no[alcohol],
    "Exercise Hours Per Week": exercise_hrs,
    "Diet": diet_map[diet],
    "Previous Heart Problems": yes_no[prev_heart_problems],
    "Medication Use": yes_no[medication_use],
    "Stress Level": stress_level,
    "Sedentary Hours Per Day": sedentary_hrs,
    "Income": income,
    "BMI": bmi,
    "Triglycerides": triglycerides,
    "Physical Activity Days Per Week": activity_days,
    "Sleep Hours Per Day": sleep_hrs,
    "Systolic_BP": systolic_bp,
    "Diastolic_BP": diastolic_bp,
}

input_df = pd.DataFrame([input_dict])[feature_order]

if st.button("Predict Risk", type="primary", use_container_width=True):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack — estimated probability: {probability:.1%}")
    else:
        st.success(f"✅ Low Risk of Heart Attack — estimated probability: {probability:.1%}")

    st.caption(
        "Note: this prediction comes from a Decision Tree trained on a public dataset "
        "for an academic project, and is not a medical diagnosis."
    )