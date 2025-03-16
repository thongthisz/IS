import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os

st.set_page_config(page_title="NN Demo", layout="wide", initial_sidebar_state="collapsed")
st.title("⏰ Neural Network Demo - Predict Wake Up Time")

# โหลดโมเดลและ Scaler
model_path = "Trained_data/wake_up_model.h5"
scaler_path = "Trained_data/scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("ยังไม่มีโมเดลหรือ Scaler กรุณาเทรนโมเดลก่อนในหน้า TrainNN.py")
    st.stop()

model = load_model(model_path, compile=False)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# ตัวเลือกการเดินทาง
transport_modes = ['Walk', 'Motorbike_Private', 'Car_Private', 'Bus', 'Train', 'None']
le_mode = LabelEncoder()
le_mode.fit(transport_modes)

# ฟอร์มกรอกข้อมูล
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
