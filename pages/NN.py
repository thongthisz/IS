import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import requests

st.set_page_config(page_title="NN Demo", layout="wide", initial_sidebar_state="collapsed")

# ========== UI Navbar ==========
st.markdown("""
    <style>
        .nav-container {
            background-color: #262730;
            padding: 15px 0;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .nav-link {
            color: white !important;
            margin: 0 20px;
            font-size: 22px;
            text-decoration: none;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        .nav-link:hover {
            background-color: #444654;
        }
    </style>

    <div class='nav-container' style='text-align: center;'>
        <a class='nav-link' href="/Data_Preparation"> Machine Info</a>
        <a class='nav-link' href="/ML"> Machine Test</a>
        <a class='nav-link' href="/Model_Explanation"> NN Info</a>
        <a class='nav-link' href="/NN"> NN Test</a>
    </div>
""", unsafe_allow_html=True)

st.title("⏰ Neural Network Demo - Predict Wake Up Time")

def load_file_from_drive(file_id, save_as):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    with open(save_as, "wb") as f:
        f.write(response.content)

file_ids = {
    "model": "1cHbx3G7CvBgeXee3qZqBLX191QrsMPH7",
    "scaler": "13ckfm3ymjJyk1xpEed0P4EWYQw6GdXa7"
}

load_file_from_drive(file_ids["model"], "wake_up_model.h5")
load_file_from_drive(file_ids["scaler"], "scaler.pkl")

model = load_model("wake_up_model.h5", compile=False)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

transport_modes = ['Walk', 'Motorbike_Private', 'Car_Private', 'Bus', 'Train', 'None']
le_mode = LabelEncoder()
le_mode.fit(transport_modes)

st.header("📋 กรอกกิจกรรมและการเดินทาง")

if 'num_transports' not in st.session_state:
    st.session_state.num_transports = 1

with st.form("predict_form"):
    class_start_time = st.text_input("เวลาเริ่มเรียน (HH:MM)", value="08:00")
    col1, col2 = st.columns(2)
    shower = col1.number_input("เวลาอาบน้ำ (นาที)", min_value=1, max_value=30, value=10)
    dressup = col2.number_input("เวลาแต่งตัว (นาที)", min_value=1, max_value=30, value=10)
    prepare = col1.number_input("เตรียมของ (นาที)", min_value=1, max_value=30, value=5)

    transports = []
    for i in range(st.session_state.num_transports):
        st.markdown(f"### 🚗 การเดินทางต่อที่ {i+1}")
        mode = st.selectbox(f"พาหนะต่อที่ {i+1}", transport_modes, key=f"mode_{i}")
        time = st.number_input(f"เวลาเดินทางต่อที่ {i+1} (นาที)", min_value=0, max_value=60, value=10, key=f"time_{i}")
        wait = st.number_input(f"เวลารอรถต่อที่ {i+1} (นาที)", min_value=0, max_value=30, value=5, key=f"wait_{i}")
        transports.append((mode, time, wait))

    add_transport = st.form_submit_button("➕ เพิ่มการเดินทาง")
    if add_transport:
        st.session_state.num_transports += 1
        st.rerun()

    buffer_time = st.number_input("เวลาเผื่อก่อนถึงเรียน (นาที)", min_value=0, max_value=60, value=10)
    predict = st.form_submit_button("ทำนายเวลาตื่น")

if predict:
    total_commute = sum(t[1] + t[2] for t in transports)
    encoded_modes = [le_mode.transform([t[0]])[0] for t in transports]

    while len(encoded_modes) < 3:
        encoded_modes.append(le_mode.transform(['None'])[0])
        transports.append(('None', 0, 0))

    input_data = pd.DataFrame([[
        shower, dressup, prepare,
        encoded_modes[0], transports[0][1], transports[0][2],
        encoded_modes[1], transports[1][1], transports[1][2],
        encoded_modes[2], transports[2][1], transports[2][2],
        total_commute, buffer_time
    ]], columns=[
        'Shower_Duration', 'DressUp_Duration', 'Prepare_Items_Duration',
        'Transport_Mode_1', 'Time_1', 'Wait_1',
        'Transport_Mode_2', 'Time_2', 'Wait_2',
        'Transport_Mode_3', 'Time_3', 'Wait_3',
        'Total_Commute_Time', 'Buffer_Time'])

    input_scaled = scaler.transform(input_data)
    wakeup_pred = model.predict(input_scaled)[0][0]
    pred_time = f"{int(wakeup_pred) // 60:02d}:{int(wakeup_pred) % 60:02d}"

    st.subheader("📈 เวลาที่ควรตื่น")
    st.write(f"### ⏰ ตื่นเวลาประมาณ: **{pred_time}**")

    leave_home_minutes = wakeup_pred + shower + dressup + prepare
    leave_time = f"{int(leave_home_minutes) // 60:02d}:{int(leave_home_minutes) % 60:02d}"
    st.write(f"🚪 ควรออกจากบ้านเวลา: **{leave_time}**")
