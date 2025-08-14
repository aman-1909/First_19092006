# eco_sandwatch_integrated.py
# Full integrated prototype: Phase 1 (all features) + Phase 2 Day 1 additions
# - Upload BEFORE/AFTER DEMs (.npy) or generate demo
# - Lat/Lon inputs, Folium map with red markers
# - Thresholded mined-zone detection, min-area filter
# - Volume calculation (m³) using pixel size
# - Visuals: Before/After/Diff/Mask (2D), Static 3D (Matplotlib), Interactive 3D (Plotly)
# - Analytics: histogram, per-site area bar
# - Downloads: CSV, GeoJSON, mask .npy, PDF with embedded images

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import folium
from streamlit_folium import st_folium
from scipy.ndimage import label, find_objects
from datetime import datetime
import math
import json
from PIL import Image as PILImage
import plotly.graph_objects as go

# ---------------- Page setup ----------------
st.set_page_config(page_title="Eco-SandWatch • Integrated Prototype", layout="wide", page_icon="🌍")
st.title("🌍 Eco-SandWatch — Integrated Prototype (Phase 1 ➜ Phase 2 Day 1)")

# ---------------- Helpers ----------------
def normalize(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def fig_to_png_bytes(fig, dpi=150):
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
    return fig_to_png_bytes(fig, dpi=160)

def plotly_surface(z, title="Surface", colorscale="Viridis"):
    z = np.asarray(z, dtype=float)
    fig = go.Figure(data=[go.Surface(z=z, colorscale=colorscale)])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X (px)", yaxis_title="Y (px)", zaxis_title="Elevation (m)"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520
    )
    return fig

def static3d_surface(z, title="Static 3D"):
    z = np.asarray(z, dtype=float)
    X, Y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, z, cmap="terrain", linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Elevation (m)")
    return fig

def generate_pdf_report(meta_rows, big_overview_png, mined_overlay_png_bytes, sites_df: pd.DataFrame):
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

    # Add main 4-panel overview figure
    story.append(Paragraph("Overview (Before / After / Diff / Mask)", styles["Heading2"]))
    story.append(RLImage(BytesIO(big_overview_png), width=470, height=150))
    story.append(Spacer(1, 10))

    # Add mined overlay
    story.append(Paragraph("Detected Mined Zones (red overlay)", styles["Heading2"]))
    story.append(RLImage(BytesIO(mined_overlay_png_bytes), width=470, height=180))
    story.append(Spacer(1, 10))

    # Sites listing
    if not sites_df.empty:
        story.append(Paragraph("Detected Sites", styles["Heading2"]))
        for _, r in sites_df.iterrows():
            story.append(Paragraph(
                f"Site {int(r['site_id'])}: Area {r['area_m2']:.1f} m² | "
                f"Mean drop {r['mean_drop_m']:.2f} m | "
                f"Lat {r['centroid_lat']:.6f}, Lon {r['centroid_lon']:.6f}",
                styles["Normal"]
            ))
            story.append(Spacer(1, 4))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ---------------- Sidebar controls ----------------
st.sidebar.header("Inputs & Controls")

# Location inputs (top-left for georeferencing + a marker point)
top_left_lat = st.sidebar.number_input("Top-left latitude (°)", value=25.612000, format="%.6f")
top_left_lon = st.sidebar.number_input("Top-left longitude (°)", value=85.158000, format="%.6f")

# Marker point for quick reference on the map
marker_lat = st.sidebar.number_input("Marker latitude (°)", value=25.594100, format="%.6f")
marker_lon = st.sidebar.number_input("Marker longitude (°)", value=85.137600, format="%.6f")

pixel_size_m = st.sidebar.number_input("Pixel size (m/pixel)", value=1.0, min_value=0.001, format="%.3f")

threshold_m = st.sidebar.slider("Elevation drop threshold (m)", 0.1, 10.0, 1.0, 0.1)
min_area_px = st.sidebar.number_input("Minimum connected area (pixels)", min_value=1, value=20, step=1)
show_bboxes = st.sidebar.checkbox("Show bounding boxes (visual)", value=False)

use_demo = st.sidebar.checkbox("Use demo DEMs (no upload)", value=False)
gen_demo = st.sidebar.button("Generate demo DEMs")

st.sidebar.markdown("---")
st.sidebar.header("Actions")
regen = st.sidebar.button("Re-run detection")
make_pdf = st.sidebar.button("📄 Build PDF report")

if regen:
    st.experimental_rerun()

# ---------------- Demo data ----------------
if gen_demo:
    h, w = 200, 300
    x = np.linspace(-3, 3, w)
    y = np.linspace(-2, 2, h)
    xv, yv = np.meshgrid(x, y)
    before_demo = 10 * np.exp(-((xv**2) + (yv**2)))
    after_demo = before_demo.copy()
    after_demo[h//2-40:h//2-5, w//2-60:w//2-20] -= 2.4
    after_demo[h//2+15:h//2+55, w//2+5:w//2+45] -= 3.1
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
            st.warning("No demo in memory. Click 'Generate demo DEMs' or upload files.")
            st.stop()
    else:
        if before_file is None or after_file is None:
            st.info("Upload both BEFORE & AFTER .npy files or enable demo.")
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

# ---------------- Core detection ----------------
diff = before - after  # positive => surface lowered => mined
raw_mask = diff > threshold_m

# Connected-component filtering
labeled, nlab = label(raw_mask)
slices = find_objects(labeled)
mask_clean = np.zeros_like(raw_mask, dtype=bool)
sites = []
site_id = 0

for i, sl in enumerate(slices, start=1):
    if sl is None:
        continue
    region = (labeled[sl] == i)
    area_px = int(np.sum(region))
    if area_px >= min_area_px:
        mask_clean[sl][region] = True
        site_id += 1
        ys, xs = np.where(labeled == i)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        area_m2 = area_px * (pixel_size_m ** 2)
        # simple lat/lon from top-left + pixel size (approximate)
        deg_per_m_lat = 1.0 / 111320.0
        deg_per_m_lon = 1.0 / (111320.0 * max(1e-6, math.cos(math.radians(top_left_lat))))
        centroid_lat = top_left_lat - (cy * pixel_size_m * deg_per_m_lat)
        centroid_lon = top_left_lon + (cx * pixel_size_m * deg_per_m_lon)
        mean_drop = float(np.mean(diff[labeled == i]))
        sites.append({
            "site_id": site_id,
            "area_px": area_px,
            "area_m2": area_m2,
            "centroid_px_x": int(cx),
            "centroid_px_y": int(cy),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "mean_drop_m": mean_drop
        })

df_sites = pd.DataFrame(sites)
total_area_m2 = float(df_sites["area_m2"].sum()) if not df_sites.empty else 0.0
total_volume_m3 = float(np.sum(np.clip(diff, 0, None)) * (pixel_size_m ** 2))

# ---------------- Tabs ----------------
tab_map, tab_visuals, tab_3d, tab_analytics, tab_export = st.tabs(
    ["Map", "2D Visuals", "3D Views", "Analytics", "Export / Report"]
)

# Map
with tab_map:
    st.subheader("Interactive Map")
    # Center roughly at raster center
    center_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0)) / 2.0
    center_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * max(1e-6, math.cos(math.radians(top_left_lat)))))) / 2.0
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")
    # raster extent rectangle
    br_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0))
    br_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * max(1e-6, math.cos(math.radians(top_left_lat))))))
    folium.Rectangle(bounds=[[top_left_lat, top_left_lon],[br_lat, br_lon]],
                     color="#006400", weight=1.2, fill=False).add_to(fmap)
    # marker (user-specified)
    folium.CircleMarker([marker_lat, marker_lon], radius=6, color="red", fill=True, fill_opacity=0.9).add_to(fmap)
    # detected sites
    for _, r in df_sites.iterrows():
        folium.CircleMarker([r["centroid_lat"], r["centroid_lon"]],
                            radius=6, color="red", fill=True, fill_opacity=0.85,
                            popup=folium.Popup(
                                f"Site {int(r['site_id'])}<br>"
                                f"Area: {r['area_m2']:.1f} m²<br>"
                                f"Mean drop: {r['mean_drop_m']:.2f} m",
                                max_width=260)
                            ).add_to(fmap)
    st_folium(fmap, width=950, height=430)

# 2D Visuals
with tab_visuals:
    st.subheader("Before / After / Difference / Mask")
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(normalize(before), cmap="terrain"); axs[0].set_title("Before DEM"); axs[0].axis("off")
    axs[1].imshow(normalize(after), cmap="terrain");  axs[1].set_title("After DEM");  axs[1].axis("off")
    axs[2].imshow(normalize(diff), cmap="RdBu");      axs[2].set_title("Diff (before - after)"); axs[2].axis("off")
    axs[3].imshow(normalize(after), cmap="terrain")
    axs[3].imshow(mask_clean, cmap=plt.cm.Reds, alpha=0.5)
    axs[3].set_title("Detected Mask"); axs[3].axis("off")
    if show_bboxes and not df_sites.empty:
        labeled_clean, _ = label(mask_clean)
        slices_c = find_objects(labeled_clean)
        for idx, sl in enumerate(slices_c, start=1):
            if sl is None: continue
            y0, y1 = sl[0].start, sl[0].stop
            x0, x1 = sl[1].start, sl[1].stop
            rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1.0, edgecolor='yellow', facecolor='none')
            axs[3].add_patch(rect)
            axs[3].text(x0, max(y0-3,0), str(idx), color='yellow', fontsize=8, weight='bold')
    st.pyplot(fig)

# 3D Views
with tab_3d:
    st.subheader("Static 3D (Matplotlib)")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(static3d_surface(before, "Before DEM — Static 3D"))
    with c2:
        st.pyplot(static3d_surface(after, "After DEM — Static 3D"))
    st.subheader("Interactive 3D (Plotly)")
    st.plotly_chart(plotly_surface(before, "Before DEM — 3D", "Terrain"), use_container_width=True)
    st.plotly_chart(plotly_surface(after, "After DEM — 3D", "Terrain"), use_container_width=True)
    st.plotly_chart(plotly_surface(diff, "Difference (m) — 3D", "RdBu"), use_container_width=True)

# Analytics
with tab_analytics:
    st.subheader("Summary Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pixels", f"{before.size:,}")
    m2.metric("Threshold (m)", f"{threshold_m:0.2f}")
    m3.metric("Detected sites", f"{0 if df_sites.empty else len(df_sites)}")
    m4.metric("Estimated Total Volume", f"{total_volume_m3:,.2f} m³")

    st.markdown("#### Depth Change Histogram (before − after)")
    fig_h, ax_h = plt.subplots(figsize=(6, 3))
    ax_h.hist(diff.flatten(), bins=60)
    ax_h.set_xlabel("Elevation drop (m)")
    ax_h.set_ylabel("Pixel count")
    st.pyplot(fig_h)

    st.markdown("#### Area by Detected Site")
    if not df_sites.empty:
        fig_a, ax_a = plt.subplots(figsize=(6, 3))
        ax_a.bar(df_sites["site_id"].astype(int), df_sites["area_m2"])
        ax_a.set_xlabel("Site ID"); ax_a.set_ylabel("Area (m²)"); ax_a.set_title("Mined area per site")
        st.pyplot(fig_a)
    else:
        st.info("No sites detected above the current threshold/min area.")

# Export / Report
with tab_export:
    st.subheader("Downloads & Report")
    # CSV (sites)
    if not df_sites.empty:
        csv_bytes = df_sites.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV (sites)", data=csv_bytes, file_name="sites.csv", mime="text/csv")

    # Mask (.npy)
    mask_buf = BytesIO()
    np.save(mask_buf, mask_clean.astype(np.uint8))
    mask_buf.seek(0)
    st.download_button("⬇️ Download detection mask (.npy)", data=mask_buf.getvalue(),
                       file_name="detection_mask.npy", mime="application/octet-stream")

    # GeoJSON (points)
    features = []
    for _, r in df_sites.iterrows():
        features.append({
            "type":"Feature",
            "properties":{"site_id": int(r["site_id"]), "area_m2": float(r["area_m2"]), "mean_drop_m": float(r["mean_drop_m"])},
            "geometry":{"type":"Point", "coordinates":[float(r["centroid_lon"]), float(r["centroid_lat"])]}
        })
    gj = {"type":"FeatureCollection", "features": features}
    st.download_button("⬇️ Download GeoJSON (centroids)", data=json.dumps(gj),
                       file_name="eco_sandwatch_sites.geojson", mime="application/geo+json")

    # Build PDF
    if make_pdf:
        overview_fig, axs = plt.subplots(1, 4, figsize=(18, 4))
        axs[0].imshow(normalize(before), cmap="terrain"); axs[0].axis("off")
        axs[1].imshow(normalize(after), cmap="terrain");  axs[1].axis("off")
        axs[2].imshow(normalize(diff), cmap="RdBu");      axs[2].axis("off")
        axs[3].imshow(normalize(after), cmap="terrain");  axs[3].imshow(mask_clean, cmap=plt.cm.Reds, alpha=0.5); axs[3].axis("off")
        overview_png = fig_to_png_bytes(overview_fig, dpi=160)

        overlay_png_bytes = mined_overlay_png(after, mask_clean)

        meta_rows = [
            ["Before file", before_name],
            ["After file",  after_name],
            ["Pixel size (m/px)", f"{pixel_size_m:.3f}"],
            ["Threshold (m)", f"{threshold_m:.2f}"],
            ["Min area (px)", str(min_area_px)],
            ["Detected sites", str(0 if df_sites.empty else len(df_sites))],
            ["Total area (m²)", f"{total_area_m2:.1f}"],
            ["Total volume (m³)", f"{total_volume_m3:.1f}"],
            ["Top-left (lat, lon)", f"{top_left_lat:.6f}, {top_left_lon:.6f}"],
            ["Marker (lat, lon)", f"{marker_lat:.6f}, {marker_lon:.6f}"],
        ]
        try:
            pdf_bytes = generate_pdf_report(meta_rows, overview_png, overlay_png_bytes, df_sites)
            st.download_button("⬇️ Download PDF Report", data=pdf_bytes,
                               file_name="EcoSandWatch_Report.pdf", mime="application/pdf")
            st.success("PDF ready.")
        except Exception as e:
            st.error(f"PDF generation error: {e}")

# Footer tip
st.caption("Tip: Adjust threshold & min area in the sidebar to tune detection. Pixel size controls volume computation.")