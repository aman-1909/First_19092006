# eco_sandwatch_pro.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import pandas as pd
import folium
from streamlit_folium import st_folium
from scipy.ndimage import label, find_objects
from datetime import datetime
import math

st.set_page_config(page_title="Eco-SandWatch Pro", page_icon="🌍", layout="wide")
st.title("🌍 Eco-SandWatch — Pro (Phase 1+)")

# ---------- Sidebar: Inputs ----------
st.sidebar.header("1) Upload DEMs")
before_file = st.sidebar.file_uploader("Before DEM (.npy)", type=["npy"])
after_file = st.sidebar.file_uploader("After DEM (.npy)", type=["npy"])

st.sidebar.header("2) Location & Pixel Size (manual)")
top_left_lat = st.sidebar.number_input("Top-left latitude (deg)", value=25.612000, format="%.6f")
top_left_lon = st.sidebar.number_input("Top-left longitude (deg)", value=85.158000, format="%.6f")
pixel_size_m = st.sidebar.number_input("Pixel size (meters/pixel)", value=1.0, format="%.3f", help="Approx scale: meters per pixel")

st.sidebar.header("3) Detection parameters")
threshold = st.sidebar.slider("Elevation drop threshold (m)", 0.1, 10.0, 1.0, 0.1)
min_area_px = st.sidebar.number_input("Minimum area (pixels)", min_value=1, value=20, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:** If your DEM has no georeference, enter the approximate top-left lat/lon and pixel size to get approximate coordinates for detected sites.")

# ---------- Main ----------
if not (before_file and after_file):
    st.info("Upload both BEFORE and AFTER `.npy` DEM files using the sidebar to start.")
    st.stop()

# Load arrays
try:
    before = np.load(before_file)
    after = np.load(after_file)
except Exception as e:
    st.error(f"Error reading .npy files: {e}")
    st.stop()

if before.shape != after.shape:
    st.error(f"Shape mismatch: before {before.shape} vs after {after.shape}. Ensure same shape.")
    st.stop()

# Compute diff and mask
diff = before - after  # positive => elevation drop
raw_mask = diff > threshold

# Remove small regions
labeled, ncomp = label(raw_mask)
slices = find_objects(labeled)
clean_mask = np.zeros_like(raw_mask, dtype=bool)
site_records = []
site_index = 0

for i, slc in enumerate(slices, start=1):
    if slc is None:
        continue
    region = (labeled[slc] == i)
    area_px = int(np.sum(region))
    if area_px >= min_area_px:
        # Accept this region
        clean_mask[slc][region] = True
        site_index += 1
        # centroid in pixel coords (y,x)
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        # approximate area in m^2
        area_m2 = area_px * (pixel_size_m ** 2)
        # approximate centroid lat/lon:
        # Convert pixel offsets to meters: (cx, cy) from top-left
        dx_m = cx * pixel_size_m
        dy_m = cy * pixel_size_m
        # degrees per meter approximations:
        deg_per_m_lat = 1.0 / 111320.0  # ~ degrees latitude per meter
        deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(top_left_lat)))  # adjusts by latitude
        centroid_lat = top_left_lat - (dy_m * deg_per_m_lat)  # downwards reduces latitude
        centroid_lon = top_left_lon + (dx_m * deg_per_m_lon)  # rightwards increases longitude
        mean_drop = float(np.mean(diff[labeled == i]))
        site_records.append({
            "site_id": site_index,
            "area_px": area_px,
            "area_m2": area_m2,
            "centroid_px_x": cx,
            "centroid_px_y": cy,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "mean_drop_m": mean_drop
        })

# ---------- Visuals: Before / After / Overlay ----------
st.subheader("Results — Before / After / Detected overlay")
col_a, col_b = st.columns([2, 1])
with col_a:
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(before, cmap="terrain")
    axs[0].set_title("Before")
    axs[1].imshow(after, cmap="terrain")
    axs[1].set_title("After")
    axs[2].imshow(after, cmap="terrain")
    axs[2].imshow(clean_mask, cmap=plt.cm.Reds, alpha=0.5)
    axs[2].set_title(f"Detected Sites: {len(site_records)}")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    st.write("**Detection Summary**")
    st.write(f"- Threshold: {threshold} m")
    st.write(f"- Min area: {min_area_px} px")
    st.write(f"- Detected sites: {len(site_records)}")
    total_area_m2 = sum([r["area_m2"] for r in site_records])
    st.write(f"- Total mined area (approx): {total_area_m2:.1f} m²")
    if len(site_records) == 0:
        st.warning("No sites reached the minimum area threshold. Try lowering threshold or min area.")
    st.markdown("---")
    if st.button("🔁 Re-run detection (use current params)"):
        st.experimental_rerun()

# ---------- Map ----------
st.subheader("Interactive map — approximate site positions")
# Center map near top-left + a small offset depending on pixel size and image dimensions
center_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0)) / 2.0
center_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * math.cos(math.radians(top_left_lat))))) / 2.0
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

# Add a small rectangle showing approximate DEM extent
# compute bottom-right lat/lon roughly
br_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0))
br_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * math.cos(math.radians(top_left_lat)))))
# rectangle corners:
folium.Rectangle(bounds=[[top_left_lat, top_left_lon], [br_lat, br_lon]], color="#006400", fill=False, weight=1.5, dash_array="5").add_to(m)

# add site markers
for rec in site_records:
    popup_html = f"""<b>Site {int(rec['site_id'])}</b><br>
    Area: {rec['area_m2']:.1f} m²<br>
    Mean drop: {rec['mean_drop_m']:.2f} m<br>
    Lat: {rec['centroid_lat']:.6f}<br>Lon: {rec['centroid_lon']:.6f}"""
    folium.Marker(location=[rec['centroid_lat'], rec['centroid_lon']],
                  popup=popup_html, icon=folium.Icon(color="red", icon="exclamation-triangle", prefix='fa')).add_to(m)

# render folium map
st_data = st_folium(m, width=700, height=400)

# ---------- Graph: mined area per site ----------
st.subheader("Mined area per detected site")
if len(site_records) > 0:
    df_sites = pd.DataFrame(site_records)
    df_sites = df_sites.sort_values("site_id")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(df_sites["site_id"].astype(int), df_sites["area_m2"])
    ax2.set_xlabel("Site ID")
    ax2.set_ylabel("Area (m²)")
    ax2.set_title("Approx Mined Area by Site")
    st.pyplot(fig2)

    # Show table
    st.dataframe(df_sites.style.format({
        "area_m2": "{:.1f}",
        "centroid_lat": "{:.6f}",
        "centroid_lon": "{:.6f}",
        "mean_drop_m": "{:.2f}"
    }))
else:
    st.info("No detected sites to graph or tabulate.")

# ---------- Downloads: CSV & PDF ----------
st.subheader("Downloads")

# CSV
if len(site_records) > 0:
    csv_buf = BytesIO()
    df_sites.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue()
    st.download_button("⬇️ Download CSV (sites)", data=csv_bytes, file_name="eco_sandwatch_sites.csv", mime="text/csv")

# PDF (report)
def make_pdf_bytes(fig_image_bytes, df_sites, before_name, after_name, threshold, min_area_px):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 50, "Eco-SandWatch — Detection Report")
    c.setFont("Helvetica", 11)
    c.drawString(50, h - 75, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, h - 95, f"Before file: {before_name}")
    c.drawString(50, h - 110, f"After file: {after_name}")
    c.drawString(50, h - 130, f"Threshold: {threshold} m   |   Min area: {min_area_px} px")
    c.drawString(50, h - 150, f"Total detected sites: {len(df_sites)}")
    c.drawString(50, h - 165, f"Total area (m²): {df_sites['area_m2'].sum():.1f}" if len(df_sites) > 0 else "Total area (m²): 0")

    # Add visualization image
    img_reader = ImageReader(BytesIO(fig_image_bytes))
    img_w = w - 100
    img_h = img_w * 0.45
    c.drawImage(img_reader, 50, h - 170 - img_h, width=img_w, height=img_h)

    # Add table of sites if present
    if len(df_sites) > 0:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, h - 50, "Detected Sites")
        c.setFont("Helvetica", 10)
        y = h - 80
        for idx, row in df_sites.iterrows():
            line = f"Site {int(row['site_id'])}: Area {row['area_m2']:.1f} m² | Mean drop {row['mean_drop_m']:.2f} m | Lat {row['centroid_lat']:.6f}, Lon {row['centroid_lon']:.6f}"
            c.drawString(50, y, line)
            y -= 14
            if y < 80:
                c.showPage()
                y = h - 50
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
else:
    # PDF without sites - basic
    empty_buf = BytesIO()
    c = canvas.Canvas(empty_buf, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 50, "Eco-SandWatch — Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, h - 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, h - 110, "No detected sites found with the current settings.")
    c.save()
    empty_buf.seek(0)
    return empty_buf.getvalue()

# Prepare fig image bytes (save the last displayed fig)
img_bytes_io = BytesIO()
fig.savefig(img_bytes_io, format="png", dpi=150, bbox_inches='tight')
img_bytes_io.seek(0)
pdf_bytes = make_pdf_bytes(img_bytes_io.getvalue(), pd.DataFrame(site_records), before_file.name, after_file.name, threshold, min_area_px)

st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="Eco_SandWatch_Report.pdf", mime="application/pdf")

st.info("Finished. Adjust threshold/min area in the sidebar to tune detection. Use demo data if needed.")