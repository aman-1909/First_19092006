import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Eco-SandWatch", layout="centered")

# App title
st.title("🚨 Eco-SandWatch")
st.subheader("Automatic Mining Detection using YOLOv8")

# File uploader
uploaded_file = st.file_uploader("📸 Upload a river image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image using PIL
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Show original image
    st.image(img_np, caption="🟡 Original Image", use_column_width=True)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_path = temp_file.name
        img.save(temp_path)

    # Load pre-trained YOLOv8n model
    with st.spinner("🔍 Detecting mining activity..."):
        model = YOLO("yolov8n.pt")  # Auto-downloads if not found
        results = model(temp_path)
        result_img = results[0].plot()  # Draw bounding boxes

    # Convert to PIL and display
    result_pil = Image.fromarray(result_img)
    st.image(result_pil, caption="🔴 Detected Mining Zones", use_column_width=True)

    # Download button
    result_pil.save("detection_result.jpg")
    with open("detection_result.jpg", "rb") as f:
        st.download_button("📥 Download Detected Image", f, file_name="mining_detection.jpg")