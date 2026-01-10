import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from twilio.rest import Client
from dotenv import load_dotenv
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="SmartLeak GIS Dashboard",
    layout="wide"
)

API_URL = "http://localhost:5000/predict"

# Load Twilio credentials
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

st.title("🌍 SmartLeak — ML-Powered Water Leakage Monitoring")

# --------------------------------------------------
# SESSION STATE (CRITICAL FIX)
# --------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
with st.form("sensor_form"):
    c1, c2, c3 = st.columns(3)

    Pressure = c1.number_input("Pressure", value=65.0)
    Flow_Rate = c1.number_input("Flow Rate", value=74.0)
    Temperature = c1.number_input("Temperature", value=97.0)

    Vibration = c2.number_input("Vibration", value=3.0)
    RPM = c2.number_input("RPM", value=2100)
    Operational_Hours = c2.number_input("Operational Hours", value=3300)

    Zone = c3.text_input("Zone", value="Zone_4")
    Block = c3.text_input("Block", value="Block_2")
    Pipe = c3.text_input("Pipe", value="Pipe_3")

    Latitude = c3.number_input("Latitude", value=25.16256, format="%.6f")
    Longitude = c3.number_input("Longitude", value=55.23802, format="%.6f")

    phone = st.text_input("📱 Alert Phone (+country code)", value="+91XXXXXXXXXX")

    submitted = st.form_submit_button("Predict Leakage")

# --------------------------------------------------
# CALL API (ON SUBMIT)
# --------------------------------------------------
if submitted:
    payload = {
        "Pressure": Pressure,
        "Flow_Rate": Flow_Rate,
        "Temperature": Temperature,
        "Vibration": Vibration,
        "RPM": RPM,
        "Operational_Hours": Operational_Hours,
        "Zone": Zone,
        "Block": Block,
        "Pipe": Pipe,
        "Latitude": Latitude,
        "Longitude": Longitude
    }

    try:
        res = requests.post(API_URL, json=payload, timeout=5)
        res.raise_for_status()
        st.session_state.prediction = {
            "response": res.json(),
            "payload": payload,
            "phone": phone
        }
    except Exception as e:
        st.error(f"API error: {e}")

# --------------------------------------------------
# DISPLAY RESULTS (PERSISTENT)
# --------------------------------------------------
if st.session_state.prediction:
    out = st.session_state.prediction["response"]
    payload = st.session_state.prediction["payload"]
    phone = st.session_state.prediction["phone"]

    leak_prob = out["leak_probability"]
    leak_flag = out["leak_detected"]

    st.subheader("📊 Prediction Result")

    c1, c2 = st.columns(2)
    c1.metric("Leak Probability", f"{leak_prob:.3f}")
    c2.metric("Leak Detected", "YES" if leak_flag else "NO")

    # --------------------------------------------------
    # GIS MAP (STABLE)
    # --------------------------------------------------
    st.subheader("📍 GIS Visualization")

    m = folium.Map(
        location=[payload["Latitude"], payload["Longitude"]],
        zoom_start=16
    )

    marker_color = "red" if leak_flag else "green"

    folium.CircleMarker(
        location=[payload["Latitude"], payload["Longitude"]],
        radius=10,
        color=marker_color,
        fill=True,
        fill_opacity=0.85,
        popup=f"Leak: {leak_flag}<br>Probability: {leak_prob:.3f}"
    ).add_to(m)

    st_folium(m, width=850, height=450)

    # --------------------------------------------------
    # SMS ALERT (ONLY IF LEAK)
    # --------------------------------------------------
    if leak_flag and phone:
        st.subheader("🚨 Alert System")

        if st.button("Send SMS Alert"):
            message = (
                "🚨 SMARTLEAK ALERT 🚨\n"
                "Water leakage detected.\n"
                f"Probability: {leak_prob:.3f}\n"
                f"Location: ({payload['Latitude']}, {payload['Longitude']})"
            )

            twilio_client.messages.create(
                body=message,
                from_=TWILIO_PHONE,
                to=phone
            )

            st.success("SMS alert sent successfully.")
    elif not leak_flag:
        st.success("✅ System operating normally — no alert required.")
