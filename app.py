import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Eco-SandWatch", layout="centered")
st.title("🌍 Eco-SandWatch Prototype")
st.subheader("Report & Detect Illegal Sand Mining")

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

st.markdown("### 📊 DEM Elevation Change & Mining Detection")

try:
    # --- Load DEM Data ---
    before = np.load("before_testing.npy")
    after = np.load("after_mining.npy")

    # --- Calculate Difference ---
    diff = before - after

    # --- Set threshold for significant change ---
    threshold = 1.0  # meters
    mining_mask = diff > threshold

    # --- Create overlay image ---
    detected = np.copy(after)
    overlay = np.zeros((after.shape[0], after.shape[1], 3), dtype=np.uint8)

    # Mark mining areas in red
    overlay[mining_mask] = [255, 0, 0]  # Red color

    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(before, cmap="terrain")
    axs[0].set_title("Before Mining")
    axs[0].axis("off")

    axs[1].imshow(after, cmap="terrain")
    axs[1].set_title("After Mining")
    axs[1].axis("off")

    axs[2].imshow(after, cmap="terrain")
    axs[2].imshow(overlay, alpha=0.5)  # Semi-transparent overlay
    axs[2].set_title("Detected Mining Areas")
    axs[2].axis("off")

    st.pyplot(fig)

except FileNotFoundError:
    st.error("❌ DEM files not found. Please upload before_testing.npy and after_mining.npy.")

# --- Report Button ---
if st.button("🚨 Report Illegal Mining"):
    st.success(f"Alert sent for mining at ({lat}, {lon}) on {date} at {time}")

st.markdown("---")
st.caption("Prototype | Grain Saviour | Riverathon 1.0 | Made by Aman Chauhan")