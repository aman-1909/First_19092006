# app.py - Eco-SandWatch Phase 1 (Clean Build)
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

# ---------- Page setup ----------
st.set_page_config(page_title="Eco-SandWatch Phase 1", page_icon="🌍", layout="wide")
st.title("🌍 Eco-SandWatch — Phase 1 (Clean Build)")

# ---------- Sidebar: Inputs ----------
st.sidebar.header("1) Upload DEMs (.npy)")
before_file = st.sidebar.file_uploader("Upload BEFORE DEM (.npy)", type=["npy"])
after_file = st.sidebar.file_uploader("Upload AFTER DEM (.npy)", type=["npy"])

st.sidebar.header("2) Location & pixel scale (manual)")
top_left_lat = st.sidebar.number_input("Top-left latitude (°)", value=25.612000, format="%.6f")
top_left_lon = st.sidebar.number_input("Top-left longitude (°)", value=85.158000, format="%.6f")
pixel_size_m = st.sidebar.number_input("Pixel size (meters/pixel)", value=1.0, format="%.3f")

st.sidebar.header("3) Detection params")
threshold = st.sidebar.slider("Elevation drop threshold (m)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
min_area_px = st.sidebar.number_input("Minimum area (pixels)", min_value=1, value=20, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("Tips: adjust threshold & min area to tune detection. Use demo arrays if you don't have DEMs.")

# ---------- Quick demo data button ----------
if st.sidebar.button("Load demo DEMs (synthetic)"):
    # Create synthetic before/after DEMs for demo
    h, w = 200, 300
    x = np.linspace(-3, 3, w)
    y = np.linspace(-2, 2, h)
    xv, yv = np.meshgrid(x, y)
    before_demo = 10 * np.exp(-((xv**2) + (yv**2)))  # hill
    after_demo = before_demo.copy()
    # carve two holes to simulate mining
    rr1 = (slice(h//2 - 30, h//2 + 10), slice(w//2 - 40, w//2 - 5))
    rr2 = (slice(h//2 + 30, h//2 + 65), slice(w//2 + 20, w//2 + 60))
    after_demo[rr1] -= 2.5
    after_demo[rr2] -= 3.0
    # small noise
    before = before_demo + np.random.normal(0, 0.02, before_demo.shape)
    after = after_demo + np.random.normal(0, 0.02, after_demo.shape)
    st.session_state["_demo_before"] = before
    st.session_state["_demo_after"] = after
    st.success("Demo DEMs loaded — press 'Run detection' in the main area.")

# ---------- Load arrays (uploads or demo) ----------
def load_arrays():
    if before_file is not None and after_file is not None:
        try:
            b = np.load(before_file)
            a = np.load(after_file)
            name_b = before_file.name
            name_a = after_file.name
            return b, a, name_b, name_a
        except Exception as e:
            st.error(f"Error reading uploaded .npy files: {e}")
            st.stop()
    # fallback to demo
    if "_demo_before" in st.session_state and "_demo_after" in st.session_state:
        return st.session_state["_demo_before"], st.session_state["_demo_after"], "demo_before.npy", "demo_after.npy"
    return None, None, None, None

before, after, before_name, after_name = load_arrays()
if before is None or after is None:
    st.info("Upload both BEFORE and AFTER .npy DEM files in the sidebar, or load demo DEMs.")
    st.stop()

# ---------- Validate shapes ----------
if before.shape != after.shape:
    st.error(f"Shape mismatch: before {before.shape} vs after {after.shape}. They must be the same shape.")
    st.stop()

# ---------- Detection ----------
diff = before - after  # positive indicates elevation drop
raw_mask = diff > threshold

# connected components to prune small patches
labeled, num_features = label(raw_mask)
slices = find_objects(labeled)
clean_mask = np.zeros_like(raw_mask, dtype=bool)
sites = []
site_counter = 0

for i, sl in enumerate(slices, start=1):
    if sl is None:
        continue
    region = (labeled[sl] == i)
    area_px = int(np.sum(region))
    if area_px >= min_area_px:
        # accept region
        clean_mask[sl][region] = True
        site_counter += 1
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        area_m2 = area_px * (pixel_size_m ** 2)
        # degrees per meter approximations
        deg_per_m_lat = 1.0 / 111320.0
        deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(top_left_lat)))
        centroid_lat = top_left_lat - (cy * pixel_size_m * deg_per_m_lat)
        centroid_lon = top_left_lon + (cx * pixel_size_m * deg_per_m_lon)
        mean_drop = float(np.mean(diff[labeled == i]))
        sites.append({
            "site_id": site_counter,
            "area_px": area_px,
            "area_m2": area_m2,
            "centroid_px_x": cx,
            "centroid_px_y": cy,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "mean_drop_m": mean_drop
        })

# ---------- Visual: side-by-side ----------
st.subheader("Visuals: Before / After / Detected overlay")
left_col, right_col = st.columns([2, 1])
with left_col:
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(before, cmap="terrain")
    axs[0].set_title("Before DEM")
    axs[1].imshow(after, cmap="terrain")
    axs[1].set_title("After DEM")
    axs[2].imshow(after, cmap="terrain")
    axs[2].imshow(clean_mask, cmap=plt.cm.Reds, alpha=0.5)
    axs[2].set_title(f"Detected Sites: {len(sites)}")
    # draw bounding boxes and numbers
    labeled_clean, n_clean = label(clean_mask)
    slices_clean = find_objects(labeled_clean)
    for idx, sl in enumerate(slices_clean, start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.2, edgecolor='yellow', facecolor='none')
        axs[2].add_patch(rect)
        axs[2].text(x0, max(y0 - 4, 0), str(idx), color='yellow', fontsize=8, weight='bold')
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

with right_col:
    st.markdown("**Detection Summary**")
    st.markdown(f"- Threshold: **{threshold} m**")
    st.markdown(f"- Min area: **{min_area_px} px**")
    st.markdown(f"- Detected sites: **{len(sites)}**")
    total_area = sum([s["area_m2"] for s in sites])
    st.markdown(f"- Total approx mined area: **{total_area:.1f} m²**")
    if len(sites) == 0:
        st.warning("No sites detected. Try lowering the threshold or the min area.")
    if st.button("🔁 Re-run detection (refresh)"):
        st.experimental_rerun()

# ---------- Interactive folium map ----------
st.subheader("Interactive map (approximate positions)")
center_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0)) / 2.0
center_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * math.cos(math.radians(top_left_lat))))) / 2.0
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

# DEM extent rectangle
br_lat = top_left_lat - (before.shape[0] * pixel_size_m * (1.0 / 111320.0))
br_lon = top_left_lon + (before.shape[1] * pixel_size_m * (1.0 / (111320.0 * math.cos(math.radians(top_left_lat)))))
folium.Rectangle(bounds=[[top_left_lat, top_left_lon], [br_lat, br_lon]],
                 color="#2E8B57", fill=False, weight=1.5, dash_array='5').add_to(m)

for s in sites:
    popup_html = (f"<b>Site {int(s['site_id'])}</b><br>"
                  f"Area: {s['area_m2']:.1f} m²<br>"
                  f"Mean drop: {s['mean_drop_m']:.2f} m<br>"
                  f"Lat: {s['centroid_lat']:.6f}<br>Lon: {s['centroid_lon']:.6f}")
    folium.Marker(location=[s["centroid_lat"], s["centroid_lon"]],
                  popup=popup_html,
                  icon=folium.Icon(color="red", icon="exclamation-triangle", prefix='fa')).add_to(m)

# show map
st_data = st_folium(m, width=900, height=420)

# ---------- Graph & table ----------
st.subheader("Mined area by site")
if len(sites) > 0:
    df_sites = pd.DataFrame(sites).sort_values("site_id")
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.bar(df_sites["site_id"].astype(int), df_sites["area_m2"])
    ax2.set_xlabel("Site ID")
    ax2.set_ylabel("Area (m²)")
    ax2.set_title("Approx Mined Area by Site")
    st.pyplot(fig2)
    st.dataframe(df_sites[['site_id','area_px','area_m2','centroid_lat','centroid_lon','mean_drop_m']].rename(columns={
        'site_id':'Site ID','area_px':'Area (px)','area_m2':'Area (m²)','centroid_lat':'Lat','centroid_lon':'Lon','mean_drop_m':'Mean drop (m)'
    }).style.format({'Area (m²)':'{:.1f}','Lat':'{:.6f}','Lon':'{:.6f}','Mean drop (m)':'{:.2f}'}))
else:
    st.info("No detected sites to show chart/table.")

# ---------- Downloads: CSV ----------
st.subheader("Downloads")
if len(sites) > 0:
    csv_buf = BytesIO()
    df_sites.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download CSV (detected sites)", data=csv_buf.getvalue(),
                       file_name="eco_sandwatch_sites.csv", mime="text/csv")

# ---------- PDF report ----------
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

    # add visualization image
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

# prepare image for PDF
imgbuf = BytesIO()
fig.savefig(imgbuf, format='png', dpi=150, bbox_inches='tight')
imgbuf.seek(0)
pdf_bytes = build_pdf(imgbuf.getvalue(), pd.DataFrame(sites), before_name, after_name, threshold, min_area_px)

st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="Eco_SandWatch_Report.pdf", mime="application/pdf")

st.success("Phase 1 clean build complete. Tweak detection parameters in sidebar to refine results.")