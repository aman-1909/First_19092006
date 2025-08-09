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

# ---------------- Page config ----------------
st.set_page_config(page_title="Eco-SandWatch Final", page_icon="🌍", layout="wide")
st.title("🌍 Eco-SandWatch — Final (Interactive)")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Upload DEMs")
before_file = st.sidebar.file_uploader("Before DEM (.npy)", type=["npy"])
after_file = st.sidebar.file_uploader("After DEM (.npy)", type=["npy"])

st.sidebar.header("Location (manual) — top-left of DEM")
top_left_lat = st.sidebar.number_input("Top-left latitude (°)", value=25.612000, format="%.6f")
top_left_lon = st.sidebar.number_input("Top-left longitude (°)", value=85.158000, format="%.6f")
pixel_size_m = st.sidebar.number_input("Pixel size (m/pixel)", value=1.0, format="%.3f")

st.sidebar.header("Detection Parameters")
threshold = st.sidebar.slider("Elevation drop threshold (m)", 0.1, 20.0, 1.0, 0.1)
min_area_px = st.sidebar.number_input("Minimum area (pixels)", min_value=1, value=20, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Notes:")
st.sidebar.write("- If DEM has no geo info, enter the approximate top-left lat/lon and pixel size.")
st.sidebar.write("- Adjust threshold/min area to tune detection.")

# ---------------- Validate inputs ----------------
if not (before_file and after_file):
    st.info("Upload both BEFORE and AFTER `.npy` DEM files from the sidebar to begin.")
    st.stop()

# ---------------- Load arrays ----------------
try:
    before = np.load(before_file)
    after = np.load(after_file)
except Exception as e:
    st.error(f"Failed to read .npy files: {e}")
    st.stop()

if before.shape != after.shape:
    st.error(f"Shape mismatch: before {before.shape} vs after {after.shape}. Both must match.")
    st.stop()

# ---------------- Compute difference and mask ----------------
diff = before - after  # positive = elevation drop
raw_mask = diff > threshold

# Remove small regions using connected components
labeled, ncomponents = label(raw_mask)
slices = find_objects(labeled)
clean_mask = np.zeros_like(raw_mask, dtype=bool)
site_records = []
site_id = 0

for i, slc in enumerate(slices, start=1):
    if slc is None:
        continue
    region = (labeled[slc] == i)
    area_px = int(np.sum(region))
    if area_px >= min_area_px:
        # Accept region
        clean_mask[slc][region] = True
        site_id += 1
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        area_m2 = area_px * (pixel_size_m ** 2)
        # pixel to approximate degrees
        deg_per_m_lat = 1.0 / 111320.0
        deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(top_left_lat)))
        centroid_lat = top_left_lat - (cy * pixel_size_m * deg_per_m_lat)
        centroid_lon = top_left_lon + (cx * pixel_size_m * deg_per_m_lon)
        mean_drop = float(np.mean(diff[labeled == i]))
        site_records.append({
            "site_id": site_id,
            "area_px": area_px,
            "area_m2": area_m2,
            "centroid_px_x": cx,
            "centroid_px_y": cy,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "mean_drop_m": mean_drop
        })

# ---------------- Visualization: Before / After / Overlay ----------------
st.subheader("Before / After / Detected Overlay")
col1, col2 = st.columns([2, 1])
with col1:
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(before, cmap="terrain")
    axs[0].set_title("Before DEM")
    axs[1].imshow(after, cmap="terrain")
    axs[1].set_title("After DEM")
    axs[2].imshow(after, cmap="terrain")
    axs[2].imshow(clean_mask, cmap=plt.cm.Reds, alpha=0.5)
    axs[2].set_title(f"Detected Sites: {len(site_records)}")
    # draw bounding boxes and numbers
    labeled2, n2 = label(clean_mask)
    slices2 = find_objects(labeled2)
    for idx, sl in enumerate(slices2, start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.5, edgecolor='yellow', facecolor='none')
        axs[2].add_patch(rect)
        axs[2].text(x0, max(y0 - 3, 0), str(idx), color='yellow', fontsize=8, weight='bold')
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("**Detection Summary**")
    st.markdown(f"- Threshold: **{threshold} m**")
    st.markdown(f"- Min area: **{min_area_px} px**")
    st.markdown(f"- Detected sites: **{len(site_records)}**")
    total_area = sum([r["area_m2"] for r in site_records])
    st.markdown(f"- Total approx mined area: **{total_area:.1f} m²**")
    if len(site_records) == 0:
        st.warning("No detected sites meet the minimum area threshold. Try lowering threshold or min area.")
    if st.button("🔁 Re-run detection (refresh)"):
        st.experimental_rerun()

# ---------------- Interactive map ----------------
st.subheader("Interactive Map (approximate positions)")
# approximate center of DEM
center_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0)) / 2.0
center_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * math.cos(math.radians(top_left_lat))))) / 2.0
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

# add DEM extent rectangle
br_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0))
br_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * math.cos(math.radians(top_left_lat)))))
folium.Rectangle(bounds=[[top_left_lat, top_left_lon], [br_lat, br_lon]],
                 color="#2E8B57", fill=False, weight=1.5, dash_array='5').add_to(m)

# add markers for sites
for rec in site_records:
    popup = folium.Popup(f"<b>Site {int(rec['site_id'])}</b><br>"
                         f"Area: {rec['area_m2']:.1f} m²<br>"
                         f"Mean drop: {rec['mean_drop_m']:.2f} m<br>"
                         f"Lat: {rec['centroid_lat']:.6f}<br>Lon: {rec['centroid_lon']:.6f}",
                         max_width=300)
    folium.Marker(location=[rec['centroid_lat'], rec['centroid_lon']],
                  popup=popup,
                  icon=folium.Icon(color="red", icon="exclamation-triangle", prefix='fa')).add_to(m)

st_folium(m, width=800, height=400)

# ---------------- Graph: mined area per site ----------------
st.subheader("Mined Area by Site")
if len(site_records) > 0:
    df_sites = pd.DataFrame(site_records)
    df_sites = df_sites.sort_values("site_id")
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.bar(df_sites["site_id"].astype(int), df_sites["area_m2"])
    ax2.set_xlabel("Site ID")
    ax2.set_ylabel("Area (m²)")
    ax2.set_title("Approx Mined Area by Site")
    st.pyplot(fig2)
    st.dataframe(df_sites.pipe(lambda d: d[['site_id','area_px','area_m2','centroid_lat','centroid_lon','mean_drop_m']]).rename(columns={
        'site_id':'Site ID','area_px':'Area (px)','area_m2':'Area (m²)','centroid_lat':'Lat','centroid_lon':'Lon','mean_drop_m':'Mean drop (m)'}).style.format({
            'Area (m²)': '{:.1f}',
            'Lat': '{:.6f}',
            'Lon': '{:.6f}',
            'Mean drop (m)': '{:.2f}'
        }))
else:
    st.info("No detected sites to display chart or table.")

# ---------------- Downloads: CSV & PDF ----------------
st.subheader("Downloads")

if len(site_records) > 0:
    # CSV
    csv_buffer = BytesIO()
    df_sites.to_csv(csv_buffer, index=False)
    st.download_button("⬇️ Download CSV (detected sites)", data=csv_buffer.getvalue(),
                       file_name="eco_sandwatch_sites.csv", mime="text/csv")

# Create an image of the visualization for PDF
img_buf = BytesIO()
fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
img_buf.seek(0)
img_bytes = img_buf.getvalue()

def build_pdf(img_bytes, df_sites, before_name, after_name, threshold, min_area_px):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 50, "Eco-SandWatch — Detection Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, h - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, h - 85, f"Before file: {before_name}")
    c.drawString(50, h - 100, f"After file: {after_name}")
    c.drawString(50, h - 115, f"Threshold: {threshold} m    Min area: {min_area_px} px")
    c.drawString(50, h - 130, f"Detected sites: {len(df_sites)}    Total area (m²): {df_sites['area_m2'].sum():.1f}" if len(df_sites)>0 else "Detected sites: 0")

    # insert image
    img_reader = ImageReader(BytesIO(img_bytes))
    img_w = w - 100
    img_h = img_w * 0.35
    c.drawImage(img_reader, 50, h - 150 - img_h, width=img_w, height=img_h)

    # add site list pages
    if len(df_sites) > 0:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, h - 50, "Detected Sites")
        c.setFont("Helvetica", 10)
        y = h - 80
        for _, r in df_sites.iterrows():
            line = f"Site {int(r['site_id'])}: Area {r['area_m2']:.1f} m² | Mean drop {r['mean_drop_m']:.2f} m | Lat {r['centroid_lat']:.6f}, Lon {r['centroid_lon']:.6f}"
            c.drawString(50, y, line)
            y -= 14
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = h - 50
    c.save()
    buf.seek(0)
    return buf.getvalue()

pdf_bytes = build_pdf(img_bytes, pd.DataFrame(site_records), before_file.name, after_file.name, threshold, min_area_px)
st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="Eco_SandWatch_Report.pdf", mime="application/pdf")

st.success("Done — adjust parameters in the sidebar to tune detection. If nothing detected, lower threshold or min area.")