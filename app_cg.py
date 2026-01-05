import streamlit as st
import pandas as pd
import joblib

# Load assets
model = joblib.load("knn_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Page Config
st.set_page_config(page_title="CardioCheck AI", page_icon="❤️", layout="wide")

# Custom CSS for a medical theme (Deep Blue and Clean White)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #004b87;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("❤️ CardioCheck AI: Heart Disease Analysis")
st.write("Fill in the patient clinical parameters below to assess heart disease risk.")

# Organize inputs into three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Patient Info")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ['M', 'F'])
    max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)

with col2:
    st.subheader("🧪 Vital Signs")
    resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col3:
    st.subheader("🩺 Clinical Tests")
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st.divider()

# Prediction Logic
if st.button("Analyze Cardiac Risk"):
    # Prepare Input
    raw_input = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
        'Sex_' + sex: 1, 'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1, 'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    
    # Prediction
    prediction = model.predict(scaled_input)[0]
    # Get probability if model supports it (KNN usually does)
    try:
        prob = model.predict_proba(scaled_input)[0][1] * 100
    except:
        prob = None

    # Result Display
    st.subheader("Analysis Results")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.error("### HIGH RISK")
            st.markdown("Immediate medical consultation is recommended.")
        else:
            st.success("### LOW RISK")
            st.markdown("Patient parameters appear within stable ranges.")

    with res_col2:
        if prob is not None:
            st.write(f"**Confidence Level:** {prob:.2f}%")
            st.progress(prob / 100)
        
        # New Feature: Feature Importance/Summary
        with st.expander("See Input Summary"):
            st.dataframe(input_df)