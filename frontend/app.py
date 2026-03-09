import streamlit as st
import requests
import numpy as np

# Set the page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        text-align: center;
        background: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("""
    Welcome to the **Heart Disease Risk Prediction System**. 
    Please input the clinical patient data via the sidebar to assess the risk of cardiovascular disease using our state-of-the-art AI model.
""")

# Sidebar inputs
st.sidebar.header("📝 Patient Clinical Data")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex (0=Female, 1=Male)", [0, 1], index=1)
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (0=False, 1=True)", [0, 1])
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, 0.1)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Flourosopy (0-4)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversable Defect)", [0, 1, 2])

    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    return features

input_data = user_input_features()

# Display current inputs
st.subheader("Selected Patient Metrics:")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Age", input_data[0])
col2.metric("Blood Pressure", f"{input_data[3]} mmHg")
col3.metric("Cholesterol", f"{input_data[4]} mg/dl")
col4.metric("Max Heart Rate", f"{input_data[7]} bpm")

st.markdown("---")

# Prediction
if st.button("🚀 Run AI Prediction Engine"):
    with st.spinner("Analyzing risk utilizing XGBoost GPU Accelerated inference..."):
        try:
            # Assuming FastAPI is running locally on 8000
            response = requests.post("http://127.0.0.1:8000/predict", json={"data": input_data})
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                
                # Render result card
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                if prediction == 1:
                    st.error("### ⚠️ High Risk of Heart Disease Detected")
                    st.markdown("The patient's clinical markers indicate a **higher likelihood** of cardiovascular issues. Immediate medical consultation is advised.")
                    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png")
                else:
                    st.success("### ✅ Low Risk of Heart Disease")
                    st.markdown("The model predicts a **low likelihood** of heart disease. Maintain a healthy lifestyle and regular checkups!")
                    st.image("https://img.icons8.com/color/96/000000/healthy-food.png")
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.warning("Communication with the AI API failed. Make sure the FastAPI backend is running via `uvicorn api.app:app --reload`.")
        except Exception as e:
            st.error(f"Error connecting to backend API: {e}")
            st.info("💡 Did you start the backend API? Run `uvicorn api.app:app` in a terminal.")

# Footer info
st.sidebar.markdown("---")
st.sidebar.info("Developed by a Machine Learning Engineer leveraging XGBoost CUDA acceleration & FastAPI.")
