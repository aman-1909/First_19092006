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

# --- File Upload for DEMs ---
before_file = st.file_uploader("📂 Upload BEFORE mining DEM (.npy)", type=["npy"])
after_file = st.file_uploader("📂 Upload AFTER mining DEM (.npy)", type=["npy"])

if before_file is not None and after_file is not None:
    try:
        before = np.load(before_file)
        after = np.load(after_file)

        # --- Calculate Difference ---
        diff = before - after
        threshold = 1.0  # meters
        mining_mask = diff > threshold

        # --- Overlay ---
        overlay = np.zeros((after.shape[0], after.shape[1], 3), dtype=np.uint8)
        overlay[mining_mask] = [255, 0, 0]  # Red

        # --- Plot ---
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(before, cmap="terrain")
        axs[0].set_title("Before Mining")
        axs[0].axis("off")

        axs[1].imshow(after, cmap="terrain")
        axs[1].set_title("After Mining")
        axs[1].axis("off")

        axs[2].imshow(after, cmap="terrain")
        axs[2].imshow(overlay, alpha=0.5)
        axs[2].set_title("Detected Mining Areas")
        axs[2].axis("off")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error reading files: {e}")
else:
    st.info("Please upload both DEM `.npy` files to continue.")

# --- Report Button ---
if st.button("🚨 Report Illegal Mining"):
    st.success(f"Alert sent for mining at ({lat}, {lon}) on {date} at {time}")

st.markdown("---")
st.caption("Prototype | Grain Saviour | Riverathon 1.0 | Made by Aman Chauhan🤗")