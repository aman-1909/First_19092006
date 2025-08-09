import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from io import BytesIO

st.title("Eco-SandWatch: Illegal Sand Mining Detection")

# Upload .npy file
uploaded_file = st.file_uploader("Upload .npy data file", type=["npy"])

if uploaded_file is not None:
    try:
        # Load the uploaded .npy data
        data = np.load(uploaded_file)
        
        st.write("### Data Preview")
        st.write(data)

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(data, marker='o')
        ax.set_title("Mining Activity Analysis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Activity Level")
        st.pyplot(fig)

        # Create PDF report
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer)
        c.setFont("Helvetica", 14)
        c.drawString(100, 800, "Eco-SandWatch Report")
        c.setFont("Helvetica", 12)
        c.drawString(100, 770, "Uploaded File: " + uploaded_file.name)
        c.drawString(100, 750, f"Data points: {len(data)}")
        c.drawString(100, 730, f"Max activity level: {np.max(data)}")
        c.drawString(100, 710, f"Min activity level: {np.min(data)}")
        c.drawString(100, 690, f"Average activity: {np.mean(data):.2f}")
        c.save()

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name="Eco_SandWatch_Report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error reading .npy file: {e}")
else:
    st.info("Please upload a .npy file to start.")