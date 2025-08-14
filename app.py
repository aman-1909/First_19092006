import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from io import BytesIO
from datetime import datetime
from scipy.ndimage import label, find_objects
import folium
from streamlit_folium import st_folium
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------------- Page setup ----------------
st.set_page_config(page_title="Eco-SandWatch • Phase 2 Day 1", layout="wide", page_icon="🌍")
st.title("🌍 Eco-SandWatch — Phase 2 Day 1 (with Phase 1 features)")

# ---------------- Helpers ----------------
def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def fig_to_png_bytes(fig, dpi=160):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

def mined_overlay_png(base_img, mask_bool):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(normalize(base_img), cmap="terrain")
    ax.imshow(mask_bool, cmap=plt.cm.Reds, alpha=0.5)
    ax.axis("off")
    return fig_to_png_bytes(fig)

def generate_pdf_report(meta_rows, overview_png, overlay_png):
    styles = getSampleStyleSheet()
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []
    story.append(Paragraph("Eco-SandWatch — Mining Detection Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Summary table
    table = Table(meta_rows, colWidths=[210, 320])
    table.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#2E8B57")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('BACKGROUND',(0,1),(-1,-1),colors.whitesmoke),
        ('VALIGN',(0,0),(-1,-1),'TOP')
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Overview (Before / After / Diff / Mask)", styles["Heading2"]))
    story.append(RLImage(BytesIO(overview_png), width=470, height=160))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Detected Mined Zones (red overlay)", styles["Heading2"]))
    story.append(RLImage(BytesIO(overlay_png), width=470, height=180))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

def static3d_surface(z, title="Static 3D"):
    z = np.asarray(z, dtype=float)
    X, Y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, z, cmap="terrain", linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation (m)")
    return fig

# ---------------- Sidebar (Phase 1 controls + new) ----------------
st.sidebar.header("Inputs & Controls")

# Location / map (Phase 1)
marker_lat = st.sidebar.number_input("Marker latitude (°)", value=25.594100, format="%.6f")
marker_lon = st.sidebar.number_input("Marker longitude (°)", value=85.137600, format="%.6f")

# Detection params
threshold_m = st.sidebar.slider("Elevation drop threshold (m)", 0.1, 10.0, 1.0, 0.1)
min_area_px = st.sidebar.number_input("Minimum connected area (pixels)", min_value=1, value=20, step=1)

# Demo or upload
use_demo = st.sidebar.checkbox("Use demo DEMs (no upload)", value=True)
gen_demo = st.sidebar.button("Generate demo DEMs")

# Actions
send_report_clicked = st.sidebar.button("🚨 Send report (placeholder)")
build_pdf_clicked = st.sidebar.button("📄 Build PDF report")

if send_report_clicked:
    st.sidebar.success("Report sent (placeholder).")

# ---------------- Demo data (Phase 1 style) ----------------
if gen_demo:
    h, w = 180, 260
    x = np.linspace(-3, 3, w)
    y = np.linspace(-2, 2, h)
    xv, yv = np.meshgrid(x, y)
    before_demo = 10 * np.exp(-((xv**2) + (yv**2)))
    after_demo = before_demo.copy()
    # two pits
    after_demo[h//2-35:h//2-5,  w//2-60:w//2-25] -= 2.2
    after_demo[h//2+10:h//2+50, w//2+10:w//2+45] -= 3.0
    before_demo += np.random.normal(0, 0.02, before_demo.shape)
    after_demo  += np.random.normal(0, 0.02, after_demo.shape)
    st.session_state["demo_before"] = before_demo
    st.session_state["demo_after"] = after_demo
    st.sidebar.success("Demo DEMs created.")

# ---------------- File uploads ----------------
col_u1, col_u2 = st.columns(2)
with col_u1:
    before_file = None if use_demo else st.file_uploader("Upload BEFORE mining DEM (.npy)", type=["npy"])
with col_u2:
    after_file  = None if use_demo else st.file_uploader("Upload AFTER mining DEM (.npy)", type=["npy"])

# ---------------- Load arrays ----------------
def load_arrays():
    if use_demo:
        if "demo_before" in st.session_state and "demo_after" in st.session_state:
            return st.session_state["demo_before"], st.session_state["demo_after"], "demo_before.npy", "demo_after.npy"
        else:
            st.info("Using built-in quick demo because no files are uploaded. Click 'Generate demo DEMs' for a fresh sample.")
            # quick fallback demo
            H, W = 120, 180
            x = np.linspace(-2.5, 2.5, W)
            y = np.linspace(-2.0, 2.0, H)
            xv, yv = np.meshgrid(x, y)
            b = 8 * np.exp(-0.6 * (xv**2 + yv**2))
            a = b.copy()
            a[H//2-25:H//2, W//2-40:W//2-10] -= 1.6
            a[H//2+5:H//2+35, W//2+5:W//2+35] -= 2.5
            return b, a, "quick_demo_before.npy", "quick_demo_after.npy"
    else:
        if before_file is None or after_file is None:
            st.warning("Upload both BEFORE & AFTER .npy files or enable demo.")
            st.stop()
        try:
            b = np.load(before_file)
            a = np.load(after_file)
            return b, a, before_file.name, after_file.name
        except Exception as e:
            st.error(f"Error loading npy files: {e}")
            st.stop()

before, after, before_name, after_name = load_arrays()

if before.shape != after.shape:
    st.error(f"Shape mismatch: BEFORE {before.shape} vs AFTER {after.shape}")
    st.stop()

# ---------------- Mining detection (Phase 1 core) ----------------
diff = before - after  # positive means surface lowered
raw_mask = diff > threshold_m

# keep only sufficiently large connected components
labels, nlab = label(raw_mask)
slices = find_objects(labels)
mask = np.zeros_like(raw_mask, dtype=bool)
for i, sl in enumerate(slices, start=1):
    if sl is None: 
        continue
    region = (labels[sl] == i)
    if np.sum(region) >= min_area_px:
        mask[sl][region] = True

detected_cells = int(np.sum(mask))
any_detected = detected_cells > 0

# ---------------- Alert (Phase 1 behavior) ----------------
if any_detected:
    st.warning(f"⚠️ Alert: Mining activity detected in {detected_cells} pixels.")
else:
    st.success("✅ No mining detected for current settings.")

# ---------------- Tabs ----------------
tab_map, tab_visuals, tab_3d, tab_graphs, tab_report = st.tabs(
    ["Map", "2D Visuals", "3D View", "Mining Graphs", "Export / Report"]
)

# Map with red dot (Phase 1)
with tab_map:
    st.subheader("Location")
    fmap = folium.Map(location=[marker_lat, marker_lon], zoom_start=12, tiles="cartodbpositron")
    folium.CircleMarker([marker_lat, marker_lon], radius=8, color="red", fill=True, fill_opacity=0.9).add_to(fmap)
    st_folium(fmap, height=380, width=900)

# 2D Visuals (Phase 1)
with tab_visuals:
    st.subheader("Before / After / Difference / Mask")
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(normalize(before), cmap="terrain"); axs[0].set_title("Before DEM"); axs[0].axis("off")
    axs[1].imshow(normalize(after),  cmap="terrain"); axs[1].set_title("After DEM");  axs[1].axis("off")
    axs[2].imshow(diff, cmap="RdBu");                axs[2].set_title("Difference (before - after)"); axs[2].axis("off")
    axs[3].imshow(normalize(after),  cmap="terrain"); 
    axs[3].imshow(mask, cmap=plt.cm.Reds, alpha=0.5); axs[3].set_title("Detected Mask"); axs[3].axis("off")
    st.pyplot(fig)

# 3D View (Phase 2 Day 1 — static)
with tab_3d:
    st.subheader("Static 3D Surface")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(static3d_surface(before, "Before DEM — Static 3D"))
    with c2:
        st.pyplot(static3d_surface(after, "After DEM — Static 3D"))

# Mining Graphs (Phase 1 + new)
with tab_graphs:
    st.subheader("Histogram of Elevation Change")
    fig_h, ax_h = plt.subplots(figsize=(6, 3.2))
    ax_h.hist(diff.flatten(), bins=60)
    ax_h.set_xlabel("Elevation drop (m)")
    ax_h.set_ylabel("Pixel count")
    st.pyplot(fig_h)

    st.subheader("Detected Area Over Threshold")
    total_pixels = diff.size
    mined_pixels = int(np.sum(diff > threshold_m))
    fig_b, ax_b = plt.subplots(figsize=(6, 3.2))
    ax_b.bar(["Over threshold", "Below threshold"], [mined_pixels, total_pixels - mined_pixels])
    ax_b.set_ylabel("Pixels")
    st.pyplot(fig_b)

# Export / Report (Phase 1 button + PDF)
with tab_report:
    st.subheader("Summary")
    st.write(f"**Files:** {before_name} / {after_name}")
    st.write(f"**Threshold:** {threshold_m:.2f} m")
    st.write(f"**Min area filter:** {min_area_px} pixels")
    st.write(f"**Detected pixels:** {detected_cells}")

    # Build overview figure for PDF
    ov_fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(normalize(before), cmap="terrain"); axs[0].axis("off")
    axs[1].imshow(normalize(after),  cmap="terrain"); axs[1].axis("off")
    axs[2].imshow(diff, cmap="RdBu");                axs[2].axis("off")
    axs[3].imshow(normalize(after),  cmap="terrain"); axs[3].imshow(mask, cmap=plt.cm.Reds, alpha=0.5); axs[3].axis("off")
    overview_png = fig_to_png_bytes(ov_fig)

    overlay_png = mined_overlay_png(after, mask)

    # PDF meta rows
    meta_rows = [
        ["Before file", before_name],
        ["After file",  after_name],
        ["Threshold (m)", f"{threshold_m:.2f}"],
        ["Min area (px)", str(min_area_px)],
        ["Detected pixels", str(detected_cells)],
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]

    if build_pdf_clicked:
        try:
            pdf_bytes = generate_pdf_report(meta_rows, overview_png, overlay_png)
            st.download_button("⬇️ Download PDF Report", data=pdf_bytes,
                               file_name="EcoSandWatch_Report.pdf", mime="application/pdf")
            st.success("PDF ready.")
        except Exception as e:
            st.error(f"PDF generation error: {e}")

    # Report action (Phase 1 style)
    if st.button("📤 Send Report (placeholder)"):
        st.success("Report sent to authorities (placeholder).")