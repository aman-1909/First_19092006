import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os


# ---------------------- PDF GENERATION ----------------------
def create_pdf(location, volume, before_img, after_img, diff_img, graph_img, map_img):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Mining Zone Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Location: {location}", styles["Normal"]))
    story.append(Paragraph(f"Estimated Mining Volume: {volume:.2f} m³", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Add Images
    for img_buf, caption in [
        (before_img, "Before DEM"),
        (after_img, "After DEM"),
        (diff_img, "Difference Map"),
        (graph_img, "Mining Graph"),
        (map_img, "Map with Red Dot")
    ]:
        story.append(Paragraph(caption, styles["Heading3"]))
        story.append(Image(img_buf, width=400, height=300))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ---------------------- UTILS ----------------------
def fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def array_to_buf(arr, title):
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap="terrain")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    buf = fig_to_buf(fig)
    plt.close(fig)
    return buf


# ---------------------- STREAMLIT APP ----------------------
st.set_page_config(page_title="Mining Zone Detector", layout="wide")

st.title("⛏️ Mining Zone Prototype - Phase 1")

# Upload DEMs
before_file = st.file_uploader("Upload BEFORE DEM (.npy)", type=["npy"])
after_file = st.file_uploader("Upload AFTER DEM (.npy)", type=["npy"])

# Location inputs
lat = st.number_input("Latitude", value=28.6139)
lon = st.number_input("Longitude", value=77.2090)

if before_file and after_file:
    before_dem = np.load(before_file)
    after_dem = np.load(after_file)

    # Difference
    diff_dem = before_dem - after_dem

    # Mining volume
    volume = np.sum(diff_dem[diff_dem > 0]) * 1  # 1 m² per pixel

    # Display Images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(before_dem, caption="Before DEM", use_container_width=True)
    with col2:
        st.image(after_dem, caption="After DEM", use_container_width=True)
    with col3:
        st.image(diff_dem, caption="Difference Map", use_container_width=True)

    st.metric("Estimated Mining Volume (m³)", f"{volume:,.2f}")

    # Mining Graph
    fig, ax = plt.subplots()
    ax.hist(diff_dem.flatten(), bins=50, color="brown", alpha=0.7)
    ax.set_title("Mining Depth Distribution")
    ax.set_xlabel("Depth Change (m)")
    ax.set_ylabel("Pixel Count")
    st.pyplot(fig)

    # Static map with red dot
    fmap = folium.Map(location=[lat, lon], zoom_start=14)
    folium.CircleMarker(location=[lat, lon], radius=6, color="red", fill=True).add_to(fmap)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_map:
        fmap.save(tmp_map.name)
        map_static = BytesIO(tmp_map.read() if os.path.exists(tmp_map.name) else b"")

    st_folium(fmap, width=700, height=500)

    # Report button
    if st.button("📄 Download PDF Report"):
        before_buf = array_to_buf(before_dem, "Before DEM")
        after_buf = array_to_buf(after_dem, "After DEM")
        diff_buf = array_to_buf(diff_dem, "Difference Map")
        graph_buf = fig_to_buf(fig)

        # Save static map image
        import selenium.webdriver as webdriver
        from selenium.webdriver.chrome.options import Options
        from PIL import Image as PILImage

        map_img_buf = BytesIO()
        fmap.save("map_temp.html")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_window_size(800, 600)
        driver.get("file://" + os.path.abspath("map_temp.html"))
        driver.save_screenshot("map_temp.png")
        driver.quit()
        with open("map_temp.png", "rb") as f:
            map_img_buf.write(f.read())
        map_img_buf.seek(0)

        pdf_buf = create_pdf(
            f"{lat}, {lon}",
            volume,
            before_buf,
            after_buf,
            diff_buf,
            graph_buf,
            map_img_buf
        )

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_buf,
            file_name="mining_report.pdf",
            mime="application/pdf"
        )