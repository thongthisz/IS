import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Redirecting...", layout="centered")
st.write("กำลังเปลี่ยนเส้นทางไปยังหน้า Data Preparation...")

switch_page("Data_Preparation")
