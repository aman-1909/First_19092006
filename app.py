# eco_sandwatch_allin.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pandas as pd
import folium
from streamlit_folium import st_folium
from scipy.ndimage import label, find_objects
from datetime import datetime
import math
import json

st.set_page_config(page_title="Eco-SandWatch — All-in Phase 1", layout="wide", page_icon="🌍")
st.title("🌍 Eco-SandWatch — All-in Phase 1")

# ---------------- Sidebar: controls ----------------
st.sidebar.header("Input / Detection Controls")
use_demo = st.sidebar.checkbox("Use demo DEMs (no upload)", value=False)

before_file = None if use_demo else st.sidebar.file_uploader("Upload BEFORE DEM (.npy)", type=["npy"])
after_file  = None if use_demo else st.sidebar.file_uploader("Upload AFTER DEM (.npy)", type=["npy"])

top_left_lat = st.sidebar.number_input("Top-left latitude (°)", value=25.612000, format="%.6f")
top_left_lon = st.sidebar.number_input("Top-left longitude (°)", value=85.158000, format="%.6f")
pixel_size_m = st.sidebar.number_input("Pixel size (m/pixel)", value=1.0, format="%.3f")

threshold = st.sidebar.slider("Elevation drop threshold (m)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
min_area_px = st.sidebar.number_input("Minimum connected area (pixels)", min_value=1, value=20, step=1)
show_bboxes = st.sidebar.checkbox("Show bounding boxes on overlay (visual only)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("Quick demo: generate synthetic DEMs for testing.")
if st.sidebar.button("Generate demo DEMs"):
    h, w = 200, 300
    x = np.linspace(-3, 3, w)
    y = np.linspace(-2, 2, h)
    xv, yv = np.meshgrid(x, y)
    before_demo = 10 * np.exp(-((xv**2) + (yv**2)))
    after_demo = before_demo.copy()
    # two excavations
    rr1 = (slice(h//2 - 30, h//2 + 0), slice(w//2 - 50, w//2 - 10))
    rr2 = (slice(h//2 + 20, h//2 + 60), slice(w//2 + 10, w//2 + 50))
    after_demo[rr1] -= 2.5
    after_demo[rr2] -= 3.5
    # small noise
    before_arr = before_demo + np.random.normal(0, 0.02, before_demo.shape)
    after_arr  = after_demo  + np.random.normal(0, 0.02, after_demo.shape)
    st.session_state["demo_before"] = before_arr
    st.session_state["demo_after"] = after_arr
    st.sidebar.success("Demo DEMs created. Uncheck 'Use demo DEMs' to upload your own or keep checked to use demo.")

# ---------------- Load arrays ----------------
def load_arrays():
    if use_demo:
        if "demo_before" in st.session_state and "demo_after" in st.session_state:
            return st.session_state["demo_before"], st.session_state["demo_after"], "demo_before.npy", "demo_after.npy"
        else:
            st.error("Demo DEMs not generated. Click 'Generate demo DEMs' in the sidebar.")
            st.stop()
    else:
        if before_file is None or after_file is None:
            st.info("Upload both BEFORE and AFTER .npy DEM files (or enable demo).")
            st.stop()
        try:
            b = np.load(before_file)
            a = np.load(after_file)
            return b, a, before_file.name, after_file.name
        except Exception as e:
            st.error(f"Error loading .npy files: {e}")
            st.stop()

before, after, before_name, after_name = load_arrays()

if before.shape != after.shape:
    st.error(f"Shape mismatch: before {before.shape} vs after {after.shape}")
    st.stop()

# ---------------- Utilities ----------------
def normalize(arr):
    a = np.array(arr, dtype=float)
    mn, mx = np.min(a), np.max(a)
    if mx - mn == 0:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

# ---------------- Detection ----------------
diff = before - after            # positive => elevation drop
raw_mask = diff > threshold

# connected components and filter small
labeled, n = label(raw_mask)
slices = find_objects(labeled)
mask_clean = np.zeros_like(raw_mask, dtype=bool)
sites = []
site_counter = 0

for i, sl in enumerate(slices, start=1):
    if sl is None:
        continue
    region = (labeled[sl] == i)
    area_px = int(np.sum(region))
    if area_px >= min_area_px:
        mask_clean[sl][region] = True
        site_counter += 1
        ys, xs = np.where(labeled == i)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        area_m2 = area_px * (pixel_size_m ** 2)
        deg_per_m_lat = 1.0 / 111320.0
        deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(top_left_lat))) if abs(top_left_lat) < 90 else 1.0/111320.0
        centroid_lat = top_left_lat - (cy * pixel_size_m * deg_per_m_lat)
        centroid_lon = top_left_lon + (cx * pixel_size_m * deg_per_m_lon)
        mean_drop = float(np.mean(diff[labeled == i]))
        sites.append({
            "site_id": site_counter,
            "area_px": area_px,
            "area_m2": area_m2,
            "centroid_px_x": int(cx),
            "centroid_px_y": int(cy),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "mean_drop_m": mean_drop
        })

df_sites = pd.DataFrame(sites)

# ---------------- Visuals ----------------
st.header("Visual Results")
col1, col2 = st.columns([2,1])
with col1:
    fig, axs = plt.subplots(1,4, figsize=(18,4))
    axs[0].imshow(normalize(before), cmap="terrain")
    axs[0].set_title("Before DEM")
    axs[1].imshow(normalize(after), cmap="terrain")
    axs[1].set_title("After DEM")
    axs[2].imshow(normalize(diff), cmap="RdBu")
    axs[2].set_title("Difference (before - after)")
    axs[3].imshow(normalize(after), cmap="terrain")
    axs[3].imshow(mask_clean, cmap=plt.cm.Reds, alpha=0.5)
    axs[3].set_title("Detected Mask")
    # optional bboxes
    if show_bboxes and len(df_sites)>0:
        labeled_clean, _ = label(mask_clean)
        slices_c = find_objects(labeled_clean)
        for idx, sl in enumerate(slices_c, start=1):
            if sl is None: continue
            y0, y1 = sl[0].start, sl[0].stop
            x0, x1 = sl[1].start, sl[1].stop
            rect = Rectangle((x0,y0), x1-x0, y1-y0, linewidth=1.2, edgecolor='yellow', facecolor='none')
            axs[3].add_patch(rect)
            axs[3].text(x0, max(y0-3,0), str(idx), color='yellow', fontsize=8, weight='bold')
    for a in axs:
        a.axis('off')
    st.pyplot(fig)

with col2:
    st.subheader("Summary")
    st.write(f"- BEFORE file: **{before_name}**")
    st.write(f"- AFTER file: **{after_name}**")
    st.write(f"- Threshold: **{threshold} m**")
    st.write(f"- Min area: **{min_area_px} px**")
    st.write(f"- Detected sites: **{len(df_sites)}**")
    total_area_m2 = df_sites['area_m2'].sum() if len(df_sites)>0 else 0.0
    total_volume_m3 = float(np.sum(diff[diff>0]) * (pixel_size_m ** 2))
    st.write(f"- Total approx mined area: **{total_area_m2:.1f} m²**")
    st.write(f"- Total approx volume: **{total_volume_m3:.1f} m³** (assuming 1 px => {pixel_size_m**2:.2f} m²)")
    st.markdown("---")
    if st.button("Re-run detection (refresh)"):
        st.experimental_rerun()

# ---------------- Interactive map ----------------
st.header("Interactive Map")
# center approx
center_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0))/2.0
center_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * math.cos(math.radians(top_left_lat))))) / 2.0
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")
# DEM extent rectangle
br_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0))
br_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * math.cos(math.radians(top_left_lat)))))
folium.Rectangle(bounds=[[top_left_lat, top_left_lon],[br_lat, br_lon]], color="#006400", fill=False, weight=1.2).add_to(m)
# add red dots
for _, r in df_sites.iterrows():
    folium.CircleMarker(location=[r['centroid_lat'], r['centroid_lon']],
                        radius=6, color='red', fill=True, fill_opacity=0.8,
                        popup=folium.Popup(f"Site {int(r['site_id'])}<br>Area: {r['area_m2']:.1f} m²<br>Mean drop: {r['mean_drop_m']:.2f} m", max_width=260)
                       ).add_to(m)
st_folium(m, width=900, height=420)

# ---------------- Graphs ----------------
st.header("Analytics & Graphs")
colg1, colg2 = st.columns(2)
with colg1:
    st.subheader("Mined area per site")
    if len(df_sites)>0:
        fig_a, ax_a = plt.subplots(figsize=(6,3))
        ax_a.bar(df_sites['site_id'].astype(int), df_sites['area_m2'])
        ax_a.set_xlabel("Site ID"); ax_a.set_ylabel("Area (m²)"); ax_a.set_title("Area by Site")
        st.pyplot(fig_a)
    else:
        st.info("No sites detected to plot.")
with colg2:
    st.subheader("Depth change histogram")
    fig_h, ax_h = plt.subplots(figsize=(6,3))
    ax_h.hist(diff.flatten(), bins=60)
    ax_h.set_title("Distribution of (before - after)")
    st.pyplot(fig_h)

# ---------------- Downloads ----------------
st.header("Downloads & Export")

# CSV
if len(df_sites)>0:
    csv_bytes = df_sites.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV (sites)", data=csv_bytes, file_name="eco_sandwatch_sites.csv", mime="text/csv")

# mask as .npy
mask_buf = BytesIO()
np.save(mask_buf, mask_clean.astype(np.uint8))
mask_buf.seek(0)
st.download_button("⬇️ Download detection mask (.npy)", data=mask_buf.getvalue(), file_name="detection_mask.npy", mime="application/octet-stream")

# GeoJSON (points)
if len(df_sites)>0:
    features = []
    for _, r in df_sites.iterrows():
        features.append({
            "type":"Feature",
            "properties": {"site_id": int(r['site_id']), "area_m2": r['area_m2'], "mean_drop_m": r['mean_drop_m']},
            "geometry": {"type":"Point", "coordinates": [r['centroid_lon'], r['centroid_lat']]}
        })
    geojson = {"type":"FeatureCollection", "features": features}
    st.download_button("⬇️ Download GeoJSON (points)", data=json.dumps(geojson), file_name="eco_sandwatch_sites.geojson", mime="application/geo+json")

# PDF report - build images and embed
def build_pdf_report():
    story_buf = BytesIO()
    doc = SimpleDocTemplate(story_buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    # Title
    story.append(Paragraph("Eco-SandWatch — Detection Report", styles['Title']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1,8))
    # summary table
    summary_data = [
        ["Before file", before_name],
        ["After file", after_name],
        ["Threshold (m)", str(threshold)],
        ["Min area (px)", str(min_area_px)],
        ["Detected sites", str(len(df_sites))],
        ["Total area (m²)", f"{total_area_m2:.1f}"],
        ["Total volume (m³)", f"{total_volume_m3:.1f}"]
    ]
    t = Table(summary_data, colWidths=[200, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2E8B57")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('BACKGROUND',(0,1),(-1,-1),colors.whitesmoke)
    ]))
    story.append(t)
    story.append(Spacer(1,12))
    # images: create bytes for the big figure used earlier
    main_fig_bytes = fig_to_bytes(fig)
    story.append(Paragraph("Detection visualization (Before / After / Diff / Mask)", styles['Heading2']))
    story.append(RLImage(BytesIO(main_fig_bytes), width=480, height=160))
    story.append(Spacer(1,12))
    # mined mask image
    mined_fig = plt.figure(figsize=(6,3))
    ax = mined_fig.add_subplot(111)
    ax.imshow(normalize(after), cmap='terrain')
    ax.imshow(mask_clean, cmap=plt.cm.Reds, alpha=0.5)
    ax.axis('off')
    mined_bytes = fig_to_bytes(mined_fig)
    story.append(Paragraph("Mined Zones (overlay)", styles['Heading2']))
    story.append(RLImage(BytesIO(mined_bytes), width=480, height=180))
    story.append(Spacer(1,12))
    # add site list
    if len(df_sites)>0:
        story.append(Paragraph("Detected Sites", styles['Heading2']))
        # simple listing
        for _, r in df_sites.iterrows():
            story.append(Paragraph(f"Site {int(r['site_id'])}: Area {r['area_m2']:.1f} m² | Mean drop {r['mean_drop_m']:.2f} m | Lat {r['centroid_lat']:.6f}, Lon {r['centroid_lon']:.6f}", styles['Normal']))
            story.append(Spacer(1,4))
    doc.build(story)
    story_buf.seek(0)
    return story_buf.getvalue()

pdf_bytes = None
if st.button("📄 Generate & Download PDF Report"):
    try:
        pdf_bytes = build_pdf_report()
        st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="EcoSandWatch_Report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF generation error: {e}")

st.info("All-in prototype ready — tweak threshold/min area to tune detection. Ask me to add YOLOv8, GeoTIFF support, or auto email alerts next.")