import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet


# PDF generation function
def create_pdf(df_sites, fig_map=None, fig_graph=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Mining Zone Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Add site data
    for i, row in df_sites.iterrows():
        story.append(Paragraph(
            f"Site {i+1}: Lat {row['Latitude']}, Lon {row['Longitude']} - Area: {row['Area']} m²",
            styles["Normal"]
        ))
        story.append(Spacer(1, 6))

    # Map
    if fig_map is not None:
        img_buf = BytesIO()
        fig_map.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        story.append(Image(img_buf, width=400, height=300))
        story.append(Spacer(1, 12))

    # Graph
    if fig_graph is not None:
        img_buf = BytesIO()
        fig_graph.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        story.append(Image(img_buf, width=400, height=300))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer


# Streamlit UI
st.title("⛏ Mining Zone Prototype - Phase 1")

# User input for lat/lon
lat = st.number_input("Latitude", value=28.6139, format="%.6f")
lon = st.number_input("Longitude", value=77.2090, format="%.6f")

# DEM file upload
before_file = st.file_uploader("Upload BEFORE DEM (.npy)", type=["npy"])
after_file = st.file_uploader("Upload AFTER DEM (.npy)", type=["npy"])

# Dummy dataframe
df_sites = pd.DataFrame([{"Latitude": lat, "Longitude": lon, "Area": 5000}])

# Map with red dot
m = folium.Map(location=[lat, lon], zoom_start=14)
folium.CircleMarker(
    location=[lat, lon],
    radius=6,
    color='red',
    fill=True,
    fill_color='red'
).add_to(m)
st_map = st_folium(m, width=700, height=500)

# Graph if DEMs are provided
fig_graph = None
if before_file and after_file:
    before_dem = np.load(before_file)
    after_dem = np.load(after_file)
    diff = after_dem - before_dem

    fig_graph, ax = plt.subplots()
    cax = ax.imshow(diff, cmap='RdBu')
    ax.set_title("Mined Zone Difference")
    plt.colorbar(cax)

    st.pyplot(fig_graph)

# Export PDF button
if st.button("📄 Download PDF Report"):
    # Save map as matplotlib figure
    fig_map, ax_map = plt.subplots()
    ax_map.set_title("Map Location")
    ax_map.text(0.5, 0.5, "See interactive map in app", ha='center', va='center')
    plt.axis("off")

    pdf_buffer = create_pdf(df_sites, fig_map=fig_map, fig_graph=fig_graph)
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name="mining_zone_report.pdf",
        mime="application/pdf"
    )