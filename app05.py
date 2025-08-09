# eco_sandwatch_app.py (Aug 12 Pro-lite)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from scipy.ndimage import label, find_objects

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Eco-SandWatch",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Eco-SandWatch")
st.markdown("**Illegal Sand Mining Detection Prototype — Phase 1 (Pro-lite)**")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Detection Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.1, 5.0, 1.0, 0.1)
min_area_px = st.sidebar.slider("Minimum Area (pixels)", 1, 500, 50, 5)

# ---------- FILE UPLOAD ----------
st.header("📤 Upload Satellite Data (.npy)")
col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("Before Image (.npy)", type=["npy"])
with col2:
    after_file = st.file_uploader("After Image (.npy)", type=["npy"])

if before_file and after_file:
    try:
        before = np.load(before_file)
        after = np.load(after_file)

        if before.shape != after.shape:
            st.error("❌ The two files must have the same dimensions.")
        else:
            diff = before - after
            change_mask = np.abs(diff) > threshold

            # ---------- DETECTION CLEANUP ----------
            labeled_array, num_features = label(change_mask)
            for feature_num in range(1, num_features + 1):
                if np.sum(labeled_array == feature_num) < min_area_px:
                    change_mask[labeled_array == feature_num] = False

            labeled_array, num_features = label(change_mask)
            object_slices = find_objects(labeled_array)

            # ---------- PLOTS ----------
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(before, cmap="gray")
            axs[0].set_title("Before")

            axs[1].imshow(after, cmap="gray")
            axs[1].set_title("After")

            cmap_overlay = colors.ListedColormap(["none", "red"])
            axs[2].imshow(after, cmap="gray")
            axs[2].imshow(change_mask, cmap=cmap_overlay, alpha=0.5)
            axs[2].set_title(f"Detected Changes ({num_features} sites)")

            # Draw bounding boxes & numbers
            for i, sl in enumerate(object_slices, start=1):
                if sl is not None:
                    y_start, y_stop = sl[0].start, sl[0].stop
                    x_start, x_stop = sl[1].start, sl[1].stop
                    rect = Rectangle(
                        (x_start, y_start),
                        x_stop - x_start,
                        y_stop - y_start,
                        linewidth=1.5, edgecolor='yellow', facecolor='none'
                    )
                    axs[2].add_patch(rect)
                    axs[2].text(x_start, y_start - 3, str(i), color='yellow', fontsize=8, weight='bold')

            for ax in axs:
                ax.axis("off")

            st.pyplot(fig)

            # ---------- PDF REPORT ----------
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=A4)
            width, height = A4

            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, height - 50, "Eco-SandWatch Report")

            c.setFont("Helvetica", 12)
            c.drawString(50, height - 80, f"Before File: {before_file.name}")
            c.drawString(50, height - 100, f"After File: {after_file.name}")
            c.drawString(50, height - 120, f"Detection Threshold: {threshold}")
            c.drawString(50, height - 140, f"Min Area (px): {min_area_px}")
            c.drawString(50, height - 160, f"Detected Sites: {num_features}")
            c.drawString(50, height - 180, f"Total Changed Pixels: {np.sum(change_mask)}")

            # Save overlay plot to image
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format="png", dpi=150)
            img_buffer.seek(0)

            c.drawImage(ImageReader(img_buffer), 50, height - 500, width=500, preserveAspectRatio=True, mask='auto')
            c.showPage()
            c.save()

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer.getvalue(),
                file_name="Eco_SandWatch_Report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Error processing files: {e}")

else:
    st.info("Please upload both `.npy` files to start detection.")