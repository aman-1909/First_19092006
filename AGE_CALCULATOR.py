import streamlit as st
from datetime import date

st.title("Age Calculator")

today = st.date_input("Select today's date", value=date.today())
dob = st.date_input("Select your date of birth")

if today >= dob:
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    st.success(f"You are {age} years old.")
else:
    st.error("Alien hai kya be !")