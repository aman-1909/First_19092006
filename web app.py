import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Eco-SandWatch", layout="centered")

st.title("🛰️ Eco-SandWatch")
st.markdown("An AI + Satellite based illegal sand mining detection prototype using DEM analysis.")

# --- Load or upload DEM data ---
st.subheader("1. Load DEM Data")
use_sample = st.checkbox("Use sample DEM data (auto-loaded)", value=True)

if use_sample:
    before = np.load("before_testing.npy")
    after = np.load("after_mining.npy")
else:
    before_file = st.file_uploader("Upload 'Before' DEM (.npy)", type="npy")
    after_file = st.file_uploader("Upload 'After' DEM (.npy)", type="npy")
    if before_file and after_file:
        before = np.load(before_file)
        after = np.load(after_file)
    else:
        st.warning("Upload both DEM files or use sample.")
        st.stop()

# --- Compute difference and mining mask ---
difference = before - after
threshold = st.slider("Set Elevation Drop Threshold (meters)", 1.0, 10.0, 2.5)
mining_mask = difference > threshold

# --- Display Maps ---
st.subheader("2. Visual Results")

# Elevation difference map
st.markdown("**Elevation Difference Map**")
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(difference, cmap="inferno")
plt.colorbar(im1, ax=ax1)
st.pyplot(fig1)

# Mining mask
st.markdown("**Detected Mining Areas**")
fig2, ax2 = plt.subplots()
ax2.imshow(mining_mask, cmap="gray")
st.pyplot(fig2)

# --- Report Button ---
st.subheader("3. Alert Reporting")
if st.button("🚨 Report Illegal Mining"):
    st.success("Alert submitted successfully! ✅ (Simulated)")

# --- Optional: Save mining mask ---
if st.button("💾 Save Mining Mask as PNG"):
    buf = BytesIO()
    plt.imsave(buf, mining_mask, cmap="gray", format='png')
    st.download_button(label="Download Mask", data=buf.getvalue(), file_name="mining_mask.png", mime="image/png")