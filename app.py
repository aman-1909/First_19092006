import streamlit as st
import numpy as np
import cv2
import random

st.title("🛰️ Eco-SandWatch: Illegal Sand Mining Detection")

# Upload river image
uploaded_file = st.file_uploader("📂 Upload a 'river.jpg' image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is not None:
        # Resize
        img = cv2.resize(img, (640, 480))

        # Create copy of original
        before_mining = img.copy()

        # Simulate mining zones by darkening rectangles
        after_mining = img.copy()
        for _ in range(10):
            x, y = random.randint(0, 540), random.randint(0, 380)
            cv2.rectangle(after_mining, (x, y), (x+100, y+100), (0, 0, 0), -1)

        # Mining detection logic
        diff = cv2.absdiff(before_mining, after_mining)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mining_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Show all outputs
        st.subheader("🖼️ Original River Image")
        st.image(cv2.cvtColor(before_mining, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.subheader("🕳️ After Simulated Illegal Mining")
        st.image(cv2.cvtColor(after_mining, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.subheader("🚨 Detected Mining Areas")
        st.image(mining_mask, clamp=True, use_column_width=True)

    else:
        st.error("❌ Error decoding image. Try another file.")
else:
    st.info("📌 Please upload an image to begin.")