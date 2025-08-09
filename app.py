import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import io

# -----------------------------
# Helper: Detect mining zones
# -----------------------------
def detect_mining_zones(before, after, threshold=5, pixel_size=1.0):
    diff = after - before
    mask = diff < -threshold  # mining = elevation drop
    coords = np.argwhere(mask)

    if coords.size == 0:
        return [], pd.DataFrame(columns=["Site ID", "Latitude", "Longitude", "Area (m²)"])

    sites = []
    data = []
    site_id = 1

    for y, x in coords:
        lat = base_lat + (y * pixel_size / 111320)  # rough lat conversion
        lon = base_lon + (x * pixel_size / (40075000 * np.cos(np.radians(base_lat)) / 360))
        area = pixel_size ** 2
        sites.append((lat, lon))
        data.append([site_id, lat, lon, area])
        site_id += 1

    df = pd.DataFrame(data, columns=["Site ID", "Latitude", "Longitude", "Area (m²)"])
    return sites, df

# -----------------------------
# Helper: Create PDF Report
# -----------------------------
def create_pdf(df, total_area, map_img_path, graph_img_path):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Mining Detection Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Total Mined Area: {total_area:.2f} m²", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Detected Sites Map", styles["Heading2"]))
    story.append(Image(map_img_path, width=400, height=300))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Mined Area Graph", styles["Heading2"]))
    story.append(Image(graph_img_path, width=400, height=300))
    story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Phase 1 - Mining Detection Prototype")

# Inputs
base_lat = st.number_input("Base Latitude", value=0.0, format="%.6f")
base_lon = st.number_input("Base Longitude", value=0.0, format="%.6f")
pixel_size = st.number_input("Pixel Size (meters)", value=30.0)

before_file = st.file_uploader("Upload BEFORE DEM (.npy)", type=["npy"])
after_file = st.file_uploader("Upload AFTER DEM (.npy)", type=["npy"])

if before_file and after_file:
    before_dem = np.load(before_file)
    after_dem = np.load(after_file)

    st.subheader("DEM Visualizations")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(before_dem, cmap='terrain')
    axs[0].set_title("Before DEM")
    axs[1].imshow(after_dem, cmap='terrain')
    axs[1].set_title("After DEM")
    diff_map = after_dem - before_dem
    axs[2].imshow(diff_map, cmap='RdBu')
    axs[2].set_title("Difference Map")
    st.pyplot(fig)

    # Detection
    sites, df_sites = detect_mining_zones(before_dem, after_dem, threshold=5, pixel_size=pixel_size)
    total_area = df_sites["Area (m²)"].sum()

    st.subheader("Detected Sites")
    st.dataframe(df_sites)

    # Map with red dots
    st.subheader("Interactive Map")
    fmap = folium.Map(location=[base_lat, base_lon], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(fmap)
    for _, row in df_sites.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.7,
            popup=f"Site {row['Site ID']}, Area: {row['Area (m²)']:.2f} m²"
        ).add_to(marker_cluster)
    st_folium(fmap, width=700, height=500)

    # Graph
    st.subheader("Mined Area Graph")
    fig2, ax2 = plt.subplots()
    ax2.bar(df_sites["Site ID"], df_sites["Area (m²)"], color='orange')
    ax2.set_xlabel("Site ID")
    ax2.set_ylabel("Area (m²)")
    ax2.set_title("Mined Area per Site")
    st.pyplot(fig2)

    # Downloads
    st.download_button(
        "Download CSV",
        df_sites.to_csv(index=False).encode("utf-8"),
        file_name="detected_sites.csv",
        mime="text/csv"
    )

    # Save temp map snapshot
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_map:
        fmap.save(tmp_map.name.replace(".png", ".html"))
        # Optional: Use selenium to screenshot map if needed
        map_img_path = tmp_map.name  # placeholder

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_graph:
        fig2.savefig(tmp_graph.name)
        graph_img_path = tmp_graph.name

    # PDF button
    pdf_buffer = create_pdf(df_sites, total_area, map_img_path, graph_img_path)
    st.download_button(
        "Download PDF Report",
        pdf_buffer,
        file_name="mining_report.pdf",
        mime="application/pdf"
    )