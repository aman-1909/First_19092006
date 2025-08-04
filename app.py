import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import datetime
import random
import io

# Title
st.set_page_config(page_title="Eco-SandWatch", layout="centered")
st.title("🌊 Eco-SandWatch")
st.caption("Detect Illegal Sand Mining using AI + Satellite Data")

# Tabs for switching tools
tab1, tab2 = st.tabs(["🏞️ DEM Mining Detection", "🛰️ YOLOv8 Simulation"])

# ---------------------- TAB 1: DEM Analysis ----------------------
with tab1:
    st.subheader("Compare DEM Maps (Before vs After Mining)")
    
    file1 = st.file_uploader("Upload BEFORE mining .npy file", type=["npy"], key="before")
    file2 = st.file_uploader("Upload AFTER mining .npy file", type=["npy"], key="after")

    if file1 and file2:
        before = np.load(file1)
        after = np.load(file2)

        # Create mining mask
        mining_mask = np.where(before - after > 2.0, 1, 0)

        # Display mask
        plt.imshow(mining_mask, cmap='Reds')
        plt.title("Detected Mining Zones (DEM Analysis)")
        plt.axis('off')

        # Save image to display
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption="Mining Detection Map", use_column_width=True)

        # Report Button
        st.markdown("---")
        st.header("📣 Report Mining Activity")

        if st.button("🚨 Report Mining", key="dem_report"):
            fake_location = random.choice(["Ganga River - Patna", "Sone River - Aurangabad", "Kosi River - Saharsa"])
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            risk = random.choice(["Low", "Moderate", "High"])

            st.success("✅ Alert Sent to Authorities!")
            st.info(f"📍 Location: {fake_location}  \n🕒 Time: {timestamp}  \n⚠️ Risk: {risk}")
        else:
            st.write("Click the button above if you detect illegal mining.")

# ---------------------- TAB 2: YOLO Simulation ----------------------
with tab2:
    st.subheader("AI-Based Visual Detection (Simulated YOLOv8)")

    uploaded_img = st.file_uploader("Upload satellite river image", type=["jpg", "jpeg", "png"], key="img")
    
    if uploaded_img:
        img = Image.open(uploaded_img)
        draw = ImageDraw.Draw(img)

        # Fake YOLO-style bounding boxes
        boxes = [
            (50, 80, 200, 220),
            (300, 150, 450, 290),
            (150, 300, 280, 420)
        ]

        for box in boxes:
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] - 10), "Mining", fill="red")

        st.image(img, caption="Simulated Mining Detection by AI", use_column_width=True)

        st.info(f"🛑 Detected {len(boxes)} mining zones")
        risk = random.choice(["Low", "Moderate", "High"])
        st.warning(f"⚠️ Risk Level: {risk}")

        # Report button (YOLO tab)
        if st.button("🚨 Report Mining", key="yolo_report"):
            fake_location = random.choice(["River Stretch A", "River Bend B", "River Fork C"])
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ Alert Sent to Authorities!")
            st.info(f"📍 Location: {fake_location}  \n🕒 Time: {timestamp}  \n⚠️ Risk: {risk}")