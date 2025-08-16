import streamlit as st import numpy as np import matplotlib.pyplot as plt import rasterio from rasterio.plot import show from fpdf import FPDF import io

---------------------------

Helper functions

---------------------------

def load_dem(file): """Load DEM file (.npy or .tif)""" if file.name.endswith(".npy"): return np.load(file) elif file.name.endswith(".tif"): with rasterio.open(file) as src: return src.read(1) else: st.error("Unsupported file format. Please upload .npy or .tif") return None

def save_as_npy(array, filename): """Save DEM as .npy file""" buffer = io.BytesIO() np.save(buffer, array) buffer.seek(0) return buffer

def detect_mining(before, after): """Simple mining detection by difference""" diff = after - before mining_mask = diff < -1  # Threshold: drop in height indicates mining return diff, mining_mask

def generate_pdf_report(diff, mining_mask): """Generate PDF with summary of results""" buffer = io.BytesIO() pdf = FPDF() pdf.add_page() pdf.set_font("Arial", size=12)

mined_area = np.sum(mining_mask)
total_area = mining_mask.size
percent_mined = (mined_area / total_area) * 100

pdf.cell(200, 10, txt="Eco-SandWatch Mining Report", ln=True, align='C')
pdf.ln(10)
pdf.multi_cell(0, 10, txt=f"Total Area Analyzed: {total_area}\nMined Pixels: {mined_area}\nMined Percentage: {percent_mined:.2f}%")

pdf.output(buffer)
buffer.seek(0)
return buffer

---------------------------

Streamlit UI

---------------------------

st.title("🌍 Eco-SandWatch Prototype") st.write("Upload BEFORE and AFTER DEM files (.npy or .tif) to detect illegal sand mining.")

before_file = st.file_uploader("Upload BEFORE DEM", type=["npy", "tif"]) after_file = st.file_uploader("Upload AFTER DEM", type=["npy", "tif"])

if before_file and after_file: before = load_dem(before_file) after = load_dem(after_file)

if before is not None and after is not None:
    diff, mining_mask = detect_mining(before, after)

    # Show DEMs
    st.subheader("📊 Uploaded DEMs")
    col1, col2 = st.columns(2)
    with col1:
        st.image(before, caption="Before DEM", use_container_width=True)
        if st.button("⬇️ Save Before DEM as .npy"):
            buffer = save_as_npy(before, "before.npy")
            st.download_button("Download before.npy", data=buffer, file_name="before.npy")
    with col2:
        st.image(after, caption="After DEM", use_container_width=True)
        if st.button("⬇️ Save After DEM as .npy"):
            buffer = save_as_npy(after, "after.npy")
            st.download_button("Download after.npy", data=buffer, file_name="after.npy")

    # Show mining difference graph
    st.subheader("⛏️ Mining Detection Graph")
    fig, ax = plt.subplots()
    ax.imshow(mining_mask, cmap="Reds")
    ax.set_title("Detected Mining Areas (Red)")
    st.pyplot(fig)

    # Download PDF Report
    if st.button("📄 Generate PDF Report"):
        pdf_buffer = generate_pdf_report(diff, mining_mask)
        st.download_button("Download Report", data=pdf_buffer, file_name="mining_report.pdf")

    # Send alert (simple simulation)
    if st.button("🚨 Send Alert"):
        st.warning("ALERT: Illegal mining detected! Authorities notified.")

