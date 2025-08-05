import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os
import datetime
import random

st.set_page_config(page_title="Eco-SandWatch", layout="centered")
st.title("🌍 Eco-SandWatch")
st.subheader("Detect and Report Illegal Sand Mining")

# File uploader
uploaded_file = st.file_uploader("📸 Upload a river image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🟡 Uploaded Image", use_column_width=True)

    # Dummy detection result
    st.markdown("✅ **Mining activity suspected** (Dummy detection)")

    # Report button
    if st.button("⚠️ Report This to Authorities"):
        # Fake GPS
        lat = round(random.uniform(25.5, 25.7), 6)
        lon = round(random.uniform(85.0, 85.3), 6)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to CSV
        data = {
            "Timestamp": [timestamp],
            "Latitude": [lat],
            "Longitude": [lon],
            "Filename": [uploaded_file.name]
        }

        df_new = pd.DataFrame(data)

        if os.path.exists("reports.csv"):
            df_existing = pd.read_csv("reports.csv")
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv("reports.csv", index=False)
        st.success("✅ Report sent! Coordinates and image saved.")

        st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))