import streamlit as st
from streamlit_lottie import st_lottie
import requests

# --- Lottie Loader ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Page Config ---
st.set_page_config(page_title="Eco-SandWatch", layout="wide")

# --- Sidebar Navigation ---
page = st.sidebar.selectbox("📂 Choose Page", ["Overview", "Live Map", "AI Detection", "Reports", "Team"])

# --- Animation ---
lottie_alert = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_gbfwtkzw.json")

# --- Page: Overview ---
if page == "Overview":
    st.title("🌊 Eco-SandWatch: Illegal Sand Mining Detector")
    st.write("Smart detection system using LiDAR + AI + DEM Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📌 Project Summary")
        st.markdown("""
        - Detect illegal sand mining in real-time  
        - Visualize riverbed changes using LiDAR & DEM  
        - Alert local authorities with geospatial evidence  
        - AI-based detection of excavation patterns
        """)

    with col2:
        st_lottie(lottie_alert, height=200)

    st.markdown("---")

    col3, col4, col5 = st.columns(3)
    col3.metric("🚨 Active Alerts", "3", "+2")
    col4.metric("🛰️ LiDAR Scans", "12", "+4")
    col5.metric("📍 Affected Districts", "5", "+1")

    with st.expander("🔍 How does Eco-SandWatch work?"):
        st.write("""
        The system collects LiDAR elevation maps, detects unusual terrain changes with AI, and flags possible sand mining sites.
        """)

# --- Page: Live Map ---
elif page == "Live Map":
    st.title("🗺️ Live Riverbed Monitoring")
    st.map()

# --- Page: AI Detection ---
elif page == "AI Detection":
    st.title("🧠 AI-Powered Mining Detection")

    tab1, tab2 = st.tabs(["Before & After Comparison", "AI Output"])

    with tab1:
        st.image("https://i.imgur.com/8zjE4Oq.png", caption="Before Mining", width=400)
        st.image("https://i.imgur.com/Q8i5wB0.png", caption="After Mining", width=400)

    with tab2:
        st.success("AI Detected 3 Unusual Changes in the Riverbed")
        st.bar_chart([0, 2, 3, 1, 3])

# --- Page: Reports ---
elif page == "Reports":
    st.title("📑 Downloadable Reports")
    st.download_button("📥 Download Latest Report", data="Your report content here", file_name="report.pdf")

# --- Page: Team ---
elif page == "Team":
    st.title("👥 Team Grain Saviour")
    st.write("""
    - Aman Chauhan — Project Lead  
    - AI Detection by: [Your Name Here]  
    - Streamlit UI by: YOU 🚀
    """)