import streamlit as st
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Eco-SandWatch • Phase 2 (3D)", layout="wide", page_icon="🌍")
st.title("🌍 Eco-SandWatch — Phase 2 with 3D DEM Views")

# -------------------- Helpers --------------------
def normalize(arr: np.ndarray):
    arr = arr.astype(float)
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def plotly_surface(z, title="Surface", colorscale="Terrain"):
    z = z.astype(float)
    h, w = z.shape
    fig = go.Figure(data=[go.Surface(z=z, colorscale=colorscale)])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (px)", yaxis_title="Y (px)", zaxis_title="Elevation (m)"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520
    )
    return fig

def mined_overlay_png(before, diff_mask):
    # 2D overlay image for the PDF (robust & lightweight)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.imshow(normalize(before), cmap="terrain")
    ax.imshow(diff_mask, cmap=plt.cm.Reds, alpha=0.5)
    ax.set_title("Mined Zones (overlay)")
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_pdf(location_label, summary_text, overlay_png: BytesIO):
    styles = getSampleStyleSheet()
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    story = [
        Paragraph("Eco-SandWatch — Mining Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(location_label, styles["Normal"]),
        Spacer(1, 8),
        Paragraph(summary_text, styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Detected Mined Zones (red overlay):", styles["Heading3"]),
        RLImage(overlay_png, width=480, height=360),
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer

# -------------------- Sidebar Inputs --------------------
st.sidebar.header("Inputs")
lat = st.sidebar.number_input("Latitude (map marker)", value=20.5937, format="%.6f")
lon = st.sidebar.number_input("Longitude (map marker)", value=78.9629, format="%.6f")
pixel_size = st.sidebar.number_input("Pixel size (m/pixel)", value=1.0, min_value=0.01, step=0.01)
thr = st.sidebar.slider("Mining threshold (m)", 0.1, 10.0, 1.0, 0.1)

st.sidebar.markdown("---")
before_file = st.sidebar.file_uploader("Upload BEFORE DEM (.npy)", type=["npy"])
after_file  = st.sidebar.file_uploader("Upload AFTER DEM (.npy)",  type=["npy"])

# Demo data generator (optional)
if st.sidebar.button("Use small demo DEMs"):
    h, w = 120, 160
    xx, yy = np.meshgrid(np.linspace(-3,3,w), np.linspace(-2,2,h))
    base = 20 + 6*np.exp(-(xx**2 + yy**2))
    b = base + np.random.normal(0, 0.02, base.shape)
    a = b.copy()
    a[50:80, 60:95] -= 2.8  # mined patch
    st.session_state["demo_before"] = b
    st.session_state["demo_after"] = a
    st.sidebar.success("Demo DEMs loaded (use even if you didn’t upload).")

# -------------------- Load DEMs --------------------
if before_file is not None and after_file is not None:
    try:
        before = np.load(before_file)
        after = np.load(after_file)
    except Exception as e:
        st.error(f"Failed to load .npy files: {e}")
        st.stop()
elif "demo_before" in st.session_state and "demo_after" in st.session_state:
    before = st.session_state["demo_before"]
    after  = st.session_state["demo_after"]
else:
    st.info("Upload BOTH .npy DEM files (Before & After) or click 'Use small demo DEMs'.")
    st.stop()

if before.shape != after.shape:
    st.error(f"Shape mismatch: before {before.shape} vs after {after.shape}")
    st.stop()

# -------------------- Core Calculations --------------------
diff = before - after         # positive => lowered surface => mined
mined_mask = diff > thr
volume_m3 = float(np.sum(np.clip(diff, 0, None)) * (pixel_size**2))

# -------------------- Layout Tabs --------------------
tab_map, tab_3d, tab_analysis, tab_report = st.tabs(["Map", "3D Views", "Analysis", "Report"])

# Map with red dot
with tab_map:
    st.subheader("Location")
    fmap = folium.Map(location=[lat, lon], zoom_start=13, tiles="cartodbpositron")
    folium.CircleMarker([lat, lon], radius=6, color="red", fill=True, fill_opacity=0.9).add_to(fmap)
    st_folium(fmap, width=900, height=420)

# 3D Views
with tab_3d:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Before DEM — 3D")
        st.plotly_chart(plotly_surface(before, "Before DEM (3D)", "Terrain"), use_container_width=True)
    with c2:
        st.subheader("After DEM — 3D")
        st.plotly_chart(plotly_surface(after, "After DEM (3D)", "Terrain"), use_container_width=True)

    st.subheader("Difference (Before − After) — 3D")
    st.plotly_chart(plotly_surface(diff, "Difference (m) — 3D", "RdBu"), use_container_width=True)

# Analysis
with tab_analysis:
    st.subheader("Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pixels", f"{before.size:,}")
    c2.metric("Threshold (m)", f"{thr:0.2f}")
    c3.metric("Estimated Mining Volume", f"{volume_m3:,.2f} m³")

    st.markdown("#### Visual: 2D overview")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.image(normalize(before), caption="Before (normalized)", use_container_width=True)
    with col_b:
        st.image(normalize(after), caption="After (normalized)", use_container_width=True)
    with col_c:
        # Show mined mask overlay (as quick view)
        overlay = np.dstack([normalize(before)]*3)  # grayscale to RGB
        st.image(mined_mask.astype(np.uint8)*255, caption="Mined mask (thresholded)", use_container_width=True)

    st.markdown("#### Histogram of Difference (m)")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(diff.flatten(), bins=60)
    ax.set_xlabel("Before − After (m)")
    ax.set_ylabel("Pixel count")
    st.pyplot(fig)

# Report
with tab_report:
    st.subheader("Generate PDF Report")
    loc_text = f"Marker location: lat {lat:.6f}, lon {lon:.6f}"
    summary = (
        f"Pixel size: {pixel_size:.2f} m; Threshold: {thr:.2f} m; "
        f"Total pixels: {before.size:,}. "
        f"Estimated mining volume: {volume_m3:,.2f} m³ "
        f"(assuming pixel area = {pixel_size**2:.2f} m²)."
    )
    overlay_png = mined_overlay_png(before, mined_mask)
    pdf_buf = generate_pdf(loc_text, summary, overlay_png)

    st.download_button(
        "⬇️ Download PDF Report",
        data=pdf_buf,
        file_name="EcoSandWatch_Phase2_Report.pdf",
        mime="application/pdf"
    )

st.success("Ready. Upload DEMs → explore in 3D → download your report.")