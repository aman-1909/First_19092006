import streamlit as st
from datetime import date
from PIL import Image
import base64

# --- Page Config ---
st.set_page_config(page_title="Cartoon Age Calculator", page_icon="🎂", layout="centered")

# --- Cartoon Background using CSS ---
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://i.imgur.com/xW8Lph9.jpg");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# --- Title ---
st.title("🎨 Cartoon Age Calculator")

# --- Age Calculation ---
today = st.date_input("📅 Select today's date", value=date.today())
dob = st.date_input("🎂 Select your date of birth")

if today >= dob:
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    st.success(f"✅ You are {age} years old.")

    # --- Cartoon Avatar by Age ---
    if age < 13:
        img_url = "https://i.imgur.com/Mo6Mly0.png"  # child
        label = "You're a cool cartoon kid!"
    elif age < 20:
        img_url = "https://i.imgur.com/bfeZsU7.png"  # teen
        label = "You're a trendy teen cartoon!"
    elif age < 50:
        img_url = "https://i.imgur.com/0fXnKQa.png"  # adult
        label = "You're a stylish cartoon adult!"
    else:
        img_url = "https://i.imgur.com/TMKAh8Y.png"  # senior
        label = "You're a wise cartoon grandmaster!"

    st.markdown(f"### 🖼️ {label}")
    st.image(img_url, width=250)

else:
    st.error("❌ Invalid date of birth! Please select a past date.")