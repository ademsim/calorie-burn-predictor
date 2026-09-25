from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Calorie Burn Predictor")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "calorie_model.pkl", "rb"))


model = load_model()

st.title("Calorie Burn Predictor")
st.write(
    "A Random Forest model (trained on the Kaggle Calorie Expenditure dataset, Playground Series S5E5) predicts "
    "how many calories you burn during a workout from your body measurements and the workout intensity."
)

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 18, 80, 35)
    height = st.slider("Height (cm)", 120, 220, 175)
    weight = st.slider("Weight (kg)", 35, 135, 75)
with col2:
    duration = st.slider("Workout duration (minutes)", 1, 30, 15)
    heart_rate = st.slider("Average heart rate (bpm)", 65, 130, 95)
    body_temp = st.slider("Body temperature (°C)", 37.0, 41.5, 40.0, 0.1)

if st.button("Predict"):
    row = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Sex_female": int(sex == "female"),
        "Sex_male": int(sex == "male"),
    }
    X = pd.DataFrame([row])[list(model.feature_names_in_)]
    calories = float(np.expm1(model.predict(X)[0]))
    st.success(f"Estimated calories burned: **{calories:.0f} kcal**")

    durations = pd.DataFrame([{**row, "Duration": d} for d in range(1, 31)])[list(model.feature_names_in_)]
    curve = pd.Series(np.expm1(model.predict(durations)), index=range(1, 31), name="Calories vs. duration")
    st.line_chart(curve)

st.caption(
    "Model: Random Forest on log-transformed calories (validation RMSLE ≈ 0.06 on the Kaggle data). "
    "Workout duration and heart rate are by far the strongest predictors. The chart shows how the estimate "
    "changes with duration when everything else stays the same."
)
