import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="CKD Predictor", page_icon="🧬", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #00c6ff, #0072ff);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, h4 {
            color: #fff7e6;
            text-align: center;
        }
        .stButton>button {
            background: linear-gradient(to right, #00b894, #00cec9);
            color: white;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.5rem 1rem;
        }
        .stSlider > div > div > div {
            background: #ffffff33;
        }
        .stSelectbox > div > div {
            background-color: #ffffff22;
        }
        .stMetricLabel, .stMetricValue {
            color: #ffffff !important;
        }
        .lang-option {
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1040/1040230.png", width=100)
    st.title("ℹ️ About")
    st.write("This app predicts Chronic Kidney Disease (CKD) based on user input.")
    st.markdown("---")
    lang = st.selectbox("🌐 Choose Language", ["English", "हिंदी"], key="lang")
    st.markdown("---")
    st.caption("Made with ❤️ by Pranav Kamboji")

# -------------------- DATA PREP --------------------
df = pd.read_csv("kidney_disease.csv")
df.dropna(inplace=True)

encode_map = {
    'yes': 1, 'no': 0,
    'normal': 1, 'abnormal': 0,
    'present': 1, 'notpresent': 0,
    'good': 1, 'poor': 0
}
df.replace(encode_map, inplace=True)

X = df[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
       'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
       'appet', 'pe', 'ane']]
y = LabelEncoder().fit_transform(df['classification'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# -------------------- TITLE --------------------
st.markdown("""
    <h1>🎑️ Chronic Kidney Disease Predictor</h1>
    <h4>Enter patient details below to assess the risk</h4>
""", unsafe_allow_html=True)

# -------------------- FORM --------------------
with st.form("prediction_form"):
    st.markdown("### 🧕 Demographics & Basic Info")
    age = st.slider("Age", 1, 100, 45)
    bp = st.slider("Blood Pressure", 50, 180, 80)
    sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])

    st.markdown("### 💉 Clinical Values")
    col1, col2 = st.columns(2)
    with col1:
        al = st.slider("Albumin", 0, 5, 1)
        su = st.slider("Sugar", 0, 5, 0)
        bgr = st.number_input("Blood Glucose Random", 70, 490, 120)
        bu = st.number_input("Blood Urea", 1.0, 200.0, 50.0)
        sc = st.number_input("Serum Creatinine", 0.1, 15.0, 1.2)
        sod = st.number_input("Sodium", 120.0, 150.0, 135.0)
    with col2:
        pot = st.number_input("Potassium", 2.0, 7.0, 4.5)
        hemo = st.number_input("Hemoglobin", 5.0, 20.0, 13.5)
        pcv = st.number_input("Packed Cell Volume", 20.0, 60.0, 41.0)
        wc = st.number_input("WBC Count", 2000.0, 20000.0, 8500.0)
        rc = st.number_input("RBC Count", 2.0, 6.5, 5.2)

    st.markdown("### 🔬 Microscopy & Conditions")
    col3, col4, col5 = st.columns(3)
    with col3:
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    with col4:
        pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
        ba = st.selectbox("Bacteria", ["present", "notpresent"])
    with col5:
        htn = st.selectbox("Hypertension", ["yes", "no"])
        dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
        cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
        appet = st.selectbox("Appetite", ["good", "poor"])
        pe = st.selectbox("Pedal Edema", ["yes", "no"])
        ane = st.selectbox("Anemia", ["yes", "no"])

    st.text_area("🩺 Doctor's Notes", placeholder="Enter any additional clinical observations here...", key="notes")

    submitted = st.form_submit_button("🔍 Predict")

# -------------------- PREDICTION --------------------
if submitted:
    input_data = pd.DataFrame([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr,
                                 bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm,
                                 cad, appet, pe, ane]],
                               columns=['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
                                        'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                                        'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])

    input_data.replace(encode_map, inplace=True)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("---")
    st.subheader("🔔 Prediction Result")
    st.metric("Kidney Disease Risk", f"{prediction:.2f}%")

    if prediction < 20:
        st.success("🟢 Low Risk — Regular monitoring advised.")
        st.info("📌 Maintain a healthy diet, stay hydrated, and get annual checkups.")
    elif prediction < 60:
        st.warning("🟡 Moderate Risk — Recommend follow-up tests.")
        st.info("📌 Schedule a nephrologist appointment. Monitor blood sugar and pressure closely.")
    else:
        st.error("🔴 High Risk — Urgent medical attention recommended.")
        st.warning("🚨 Consider ultrasound, GFR tests, and specialist care ASAP.")

# -------------------- FOOTER --------------------
st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <big>Made with ❤️ | Kamboji Pranav The King &copy; All ways</big>
    </div>
""", unsafe_allow_html=True)
