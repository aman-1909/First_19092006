# eco_sandwatch_aug9.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import io

# Page config
st.set_page_config(page_title="Eco-SandWatch", layout="centered", page_icon="🌍")

# ---------- Header ----------
st.markdown(
    """
    # 🌍 Eco-SandWatch
    **An AI-assisted, low-cost tool to detect illegal sand mining from elevation models (DEMs).**  
    Upload *before* and *after* DEMs (.npy), pinpoint location, and automatically detect significant elevation loss.

    ---
    """
)

# ---------- Layout: Inputs ----------
st.markdown("## 📍 Location & Observation")
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=25.612000, format="%.6f", help="Enter observation latitude")
with col2:
    lon = st.number_input("Longitude", value=85.158000, format="%.6f", help="Enter observation longitude")

col3, col4 = st.columns(2)
with col3:
    date = st.date_input("Date", datetime.date.today())
with col4:
    time = st.time_input("Time", datetime.datetime.now().time())

# quick map preview
st.markdown("**Map preview**")
map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
st.map(map_data)

st.markdown("---")

# ---------- DEM Upload / Demo Generator ----------
st.markdown("## 📂 DEM Files (Before / After)")
st.caption("Upload two NumPy .npy files containing 2D elevation arrays (same shape).")

u1, u2 = st.columns(2)
with u1:
    before_file = st.file_uploader("Upload BEFORE mining DEM (.npy)", type=["npy"], key="before")
with u2:
    after_file = st.file_uploader("Upload AFTER mining DEM (.npy)", type=["npy"], key="after")

st.markdown("**Need demo data?**")
if st.button("🔁 Generate demo DEMs (synthetic)"):
    # create synthetic DEMs: before = smooth mound, after = excavation area
    h, w = 200, 300
    x = np.linspace(-3, 3, w)
    y = np.linspace(-2, 2, h)
    xv, yv = np.meshgrid(x, y)
    before_demo = 10 * np.exp(-((xv**2) + (yv**2)))
    after_demo = before_demo.copy()
    # simulate excavation: carve a patch
    cx, cy, r = w // 2 + 20, h // 2 - 10, 30
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    after_demo[mask] -= 2.5  # remove elevation
    # add slight noise
    before_demo += np.random.normal(0, 0.02, before_demo.shape)
    after_demo += np.random.normal(0, 0.02, after_demo.shape)

    # keep in session_state so app can use without upload
    st.session_state["before_demo"] = before_demo
    st.session_state["after_demo"] = after_demo
    st.success("Demo DEMs generated and loaded. Scroll to visualization.")

st.markdown("---")

# ---------- Detection Parameters ----------
st.markdown("## ⚙️ Detection Settings")
threshold = st.slider("Elevation drop threshold (meters) — mark area as mined if drop > threshold",
                      min_value=0.1, max_value=5.0, value=1.0, step=0.1)
min_area_px = st.number_input("Minimum connected area (pixels) to consider (helps reduce noise)", value=20, min_value=1)

st.markdown("---")

# ---------- Load arrays (uploaded or demo) ----------
def load_uploaded_or_demo():
    if before_file is not None and after_file is not None:
        try:
            before = np.load(before_file)
            after = np.load(after_file)
            return before, after, "uploaded"
        except Exception as e:
            st.error(f"Error reading uploaded files: {e}")
            return None, None, None
    # fallback to demo in session_state if present
    if "before_demo" in st.session_state and "after_demo" in st.session_state:
        return st.session_state["before_demo"], st.session_state["after_demo"], "demo"
    return None, None, None

before, after, source = load_uploaded_or_demo()

# ---------- Visualization & Detection ----------
if before is None or after is None:
    st.info("Please upload both .npy files or generate demo DEMs to visualize and detect mining.")
else:
    # Validate shapes
    if before.shape != after.shape:
        st.error(f"Shape mismatch: before {before.shape} vs after {after.shape}. Ensure both DEMs have same dimensions.")
    else:
        # compute difference (positive means elevation drop: before - after)
        diff = before - after

        # simple mask
        raw_mask = diff > threshold

        # optional: remove small islands (connected components)
        try:
            # lightweight connected components using scipy if available, otherwise skip
            from scipy import ndimage as ndi
            labeled, ncomp = ndi.label(raw_mask)
            sizes = ndi.sum(raw_mask, labeled, range(1, ncomp + 1))
            large_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_area_px]
            mask = np.isin(labeled, large_labels)
        except Exception:
            # scipy not available on phone; fall back to raw_mask
            mask = raw_mask

        # Prepare overlay (RGB)
        overlay = np.zeros((after.shape[0], after.shape[1], 3), dtype=np.uint8)
        overlay[mask] = [255, 0, 0]  # red

        # Plot side-by-side
        st.markdown("## 📊 Results")
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))

        axs[0].imshow(before, cmap="terrain")
        axs[0].set_title("Before")
        axs[0].axis("off")

        axs[1].imshow(after, cmap="terrain")
        axs[1].set_title("After")
        axs[1].axis("off")

        axs[2].imshow(after, cmap="terrain")
        axs[2].imshow(overlay, alpha=0.55)
        axs[2].set_title("Detected Mining Areas")
        axs[2].axis("off")

        st.pyplot(fig)

        # Summary stats
        detected_area_px = int(np.sum(mask))
        mean_drop = float(np.mean(diff[mask]) if detected_area_px > 0 else 0.0)
        st.markdown(f"**Source:** {source} DEMs &nbsp; • &nbsp; **Detected pixels:** {detected_area_px} &nbsp; • &nbsp; **Mean drop (on mask):** {mean_drop:.2f} m")

        # Option to save mask as .npy for later
        buf = io.BytesIO()
        np.save(buf, mask.astype(np.uint8))
        buf.seek(0)
        st.download_button("💾 Download detection mask (.npy)", data=buf, file_name="mining_mask.npy", mime="application/octet-stream")

st.markdown("---")

# ---------- Report button ----------
if st.button("🚨 Report Illegal Mining"):
    st.success(f"Alert sent for mining at ({lat}, {lon}) on {date} at {time}")
    st.info("Prototype alert — integrate backend (DB / WhatsApp / SMS) to send real notifications.")

st.caption("Prototype | Grain Saviour | Riverathon 1.0 | Made by Aman Chauhan")