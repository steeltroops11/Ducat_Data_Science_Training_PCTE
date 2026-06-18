import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Pulsar Star Detection",
    page_icon="⭐",
    layout="wide"
)

# ====================================
# LOAD MODEL
# ====================================
from pathlib import Path
import joblib
import streamlit as st

@st.cache_resource
def load_model():

    BASE_DIR = Path(__file__).parent

    st.write("Current Directory:", BASE_DIR)

    st.write("Files Available:")
    st.write(list(BASE_DIR.iterdir()))

    model = joblib.load(BASE_DIR / "logistic_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")

    return model, scaler

model, scaler = load_model()

# ====================================
# SESSION STATE DEFAULTS
# ====================================

defaults = {
    "mean_profile": 100.0,
    "std_profile": 45.0,
    "kurtosis_profile": 0.0,
    "skewness_profile": 0.0,
    "mean_dmsnr": 10.0,
    "std_dmsnr": 20.0,
    "kurtosis_dmsnr": 5.0,
    "skewness_dmsnr": 50.0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ====================================
# SIDEBAR
# ====================================

st.sidebar.title("📊 Model Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("""
### Model Details

- Algorithm: Logistic Regression
- Dataset: HTRU_2
- Accuracy: 97.88%
- ROC-AUC: 97.44%
- Precision: 94%
- Recall: 82%
""")

st.sidebar.info(
    "This model predicts whether an astronomical signal belongs to a Pulsar Star."
)

# ====================================
# HEADER
# ====================================

st.title("⭐ Pulsar Star Detection System")

st.markdown(
    "Machine Learning Powered Classification of Pulsar Candidates"
)

st.divider()

# ====================================
# TOP METRICS
# ====================================

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", "97.88%")
m2.metric("ROC-AUC", "97.44%")
m3.metric("Precision", "94%")
m4.metric("Recall", "82%")

st.divider()

# ====================================
# EXAMPLE BUTTON
# ====================================

if st.button("⭐ Load Example Pulsar Input"):

    st.session_state.mean_profile = 102.50
    st.session_state.std_profile = 58.88
    st.session_state.kurtosis_profile = 0.46
    st.session_state.skewness_profile = 0.14
    st.session_state.mean_dmsnr = 81.89
    st.session_state.std_dmsnr = 89.39
    st.session_state.kurtosis_dmsnr = 1.43
    st.session_state.skewness_dmsnr = 0.39

# ====================================
# INPUTS
# ====================================

st.subheader("Input Feature Values")

col1, col2 = st.columns(2)

with col1:

    mean_profile = st.number_input(
        "Mean of Integrated Profile",
        key="mean_profile"
    )

    std_profile = st.number_input(
        "Standard Deviation of Profile",
        key="std_profile"
    )

    kurtosis_profile = st.number_input(
        "Excess Kurtosis of Profile",
        key="kurtosis_profile"
    )

    skewness_profile = st.number_input(
        "Skewness of Profile",
        key="skewness_profile"
    )

with col2:

    mean_dmsnr = st.number_input(
        "Mean of DM-SNR Curve",
        key="mean_dmsnr"
    )

    std_dmsnr = st.number_input(
        "Std of DM-SNR Curve",
        key="std_dmsnr"
    )

    kurtosis_dmsnr = st.number_input(
        "Kurtosis of DM-SNR Curve",
        key="kurtosis_dmsnr"
    )

    skewness_dmsnr = st.number_input(
        "Skewness of DM-SNR Curve",
        key="skewness_dmsnr"
    )

# ====================================
# FEATURE TABLE
# ====================================

feature_df = pd.DataFrame({
    "Feature": [
        "mean_profile",
        "std_profile",
        "kurtosis_profile",
        "skewness_profile",
        "mean_dmsnr",
        "std_dmsnr",
        "kurtosis_dmsnr",
        "skewness_dmsnr"
    ],
    "Value": [
        mean_profile,
        std_profile,
        kurtosis_profile,
        skewness_profile,
        mean_dmsnr,
        std_dmsnr,
        kurtosis_dmsnr,
        skewness_dmsnr
    ]
})

with st.expander("📋 View Entered Features"):
    st.dataframe(feature_df, use_container_width=True)

# ====================================
# PREDICTION
# ====================================

if st.button("🚀 Predict", use_container_width=True):

    data = np.array([[
        mean_profile,
        std_profile,
        kurtosis_profile,
        skewness_profile,
        mean_dmsnr,
        std_dmsnr,
        kurtosis_dmsnr,
        skewness_dmsnr
    ]])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("⭐ Pulsar Star Detected")
    else:
        st.error("❌ Not a Pulsar Star")

    st.metric(
        "Probability of Pulsar",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    confidence = max(
        probability,
        1 - probability
    )

    st.metric(
        "Model Confidence",
        f"{confidence*100:.2f}%"
    )

# ====================================
# ABOUT SECTION
# ====================================

st.divider()

st.subheader("About This Project")

st.write("""
This application uses a Logistic Regression model trained on the
HTRU_2 Pulsar Star Dataset.

The model analyzes statistical characteristics extracted from
radio telescope signals and predicts whether the signal belongs
to a Pulsar Star.
""")

st.write("### Technologies Used")

st.write("""
- Python
- Streamlit
- NumPy
- Pandas
- Scikit-Learn
- Joblib
""")

# ====================================
# FOOTER
# ====================================

st.divider()

st.caption(
    "Developed by Navish Sharma | B.Tech AI & ML | Machine Learning Internship Project"
)