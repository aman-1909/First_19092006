import streamlit as st
import pandas as pd
import datetime

st.set_page_config(page_title="Eco-SandWatch", layout="centered")
st.title("🌍 Eco-SandWatch Prototype")
st.subheader("Report Illegal Sand Mining")

# --- GPS Coordinates Input ---
lat = st.number_input("📍 Enter Latitude", value=25.612, format="%.6f")
lon = st.number_input("📍 Enter Longitude", value=85.158, format="%.6f")

# --- Date and Time Inputs (Optional but useful) ---
date = st.date_input("🗓️ Date of Observation", datetime.date.today())
time = st.time_input("⏰ Time of Observation", datetime.datetime.now().time())

# --- Display map with pin ---
st.write("🗺️ Mapped Location:")
map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data)

# --- Report Button ---
if st.button("🚨 Report Illegal Mining"):
    st.success(f"Alert sent for mining at ({lat}, {lon}) on {date} at {time}")

    # Placeholder for actual backend alerting logic
    # Example: send to database, WhatsApp bot, email, etc.

st.markdown("---")
st.caption("Prototype | Grain Saviour | Riverathon 1.0 | Made by Aman Chauhan")