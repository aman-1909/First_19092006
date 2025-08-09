import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Eco-SandWatch", layout="centered")
st.title("🌍 Eco-SandWatch Prototype")
st.subheader("Report & Visualize Illegal Sand Mining")

# --- GPS Coordinates Input ---
lat = st.number_input("📍 Enter Latitude", value=25.612, format="%.6f")
lon = st.number_input("📍 Enter Longitude", value=85.158, format="%.6f")

# --- Date and Time Inputs ---
date = st.date_input("🗓️ Date of Observation", datetime.date.today())
time = st.time_input("⏰ Time of Observation", datetime.datetime.now().time())

# --- Map Display ---
st.write("🗺️ Mapped Location:")
map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data)

st.markdown("### 📊 DEM Elevation Change Visualization")

# --- Load DEM Data ---
try:
    before = np.load("before_testing.npy")
    after = np.load("after_mining.npy")

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(before, cmap="terrain")
    axs[0].set_title("Before Mining")
    axs[0].axis("off")

    axs[1].imshow(after, cmap="terrain")
    axs[1].set_title("After Mining")
    axs[1].axis("off")

    st.pyplot(fig)

except FileNotFoundError:
    st.error("❌ DEM files not found. Please upload before_testing.npy and after_mining.npy.")

# --- Report Button ---
if st.button("🚨 Report Illegal Mining"):
    st.success(f"Alert sent for mining at ({lat}, {lon}) on {date} at {time}")

st.markdown("---")
st.caption("Prototype | Grain Saviour | Riverathon 1.0 | Made by Aman Chauhan")