import streamlit as st
import numpy as np
from mining import detect_mining
import matplotlib.pyplot as plt
import datetime
import random

st.title("--Grain Patrol--")
st.write("Our aim is to Save Environment")

before_file = st.file_uploader("Upload BEFORE DEM (.npy)", type=["npy"])
after_file = st.file_uploader("Upload AFTER DEM (.npy)", type=["npy"])

if before_file and after_file:
    before = np.load(before_file)
    after = np.load(after_file)

    change, mask, volume = detect_mining(before, after)

    st.subheader("🌍 Elevation Change Heatmap")
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(change, cmap='coolwarm')
    plt.colorbar(im, ax=ax1)
    st.pyplot(fig1)

    st.subheader("⚠️ Mining Mask (Drop > 1m)")
    fig2, ax2 = plt.subplots()
    ax2.imshow(mask, cmap='gray')
    st.pyplot(fig2)

    st.success(f"✅ Estimated Volume of Sand Mined: {volume:.2f} m³")



st.markdown("---")
st.header("📣 Report Mining Activity")

# Report Button
if st.button("🚨 Report Mining"):
    # Simulated data
    fake_location = random.choice(["Ganga River - Patna", "Sone River - Aurangabad", "Kosi River - Saharsa"])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fake_risk = random.choice(["Low", "Moderate", "High"])

    # Display alert
    st.success("✅ Alert Sent to Authorities!")
    st.info(f"""
    **Location:** {fake_location}  
    **Time:** {timestamp}  
    **Risk Level:** {fake_risk}
    """)

    # Optional sound effect (works only on laptop/desktop with audio):
    # st.audio("alert_sound.mp3")

else:
    st.write("Click the button above if you detect illegal mining.")