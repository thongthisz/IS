import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="NN Info", layout="wide", initial_sidebar_state="collapsed")

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

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div style='text-align: left; margin-top: 50px; margin-left: 100px;'>
        <h2 style='font-size: 40px;'>การเตรียมข้อมูล (Data Preparation)</h2>
    </div>
    <div style='font-size:20px; text-align: left; line-height: 1.6;'>
    <ul>
    <li>ตรวจสอบข้อมูลที่ขาดหาย (Missing Values)</li>
    <li>ลบแถวที่มีข้อมูลขาด หรือเติมค่ากลาง เช่น ค่าเฉลี่ย (Mean), ค่าที่พบบ่อย (Mode)</li>
    <li>แปลงข้อมูลเวลา เช่น HH:MM → นาที</li>
    <li>แปลงข้อมูลหมวดหมู่ เช่น Transport Mode ด้วย Label Encoding</li>
    <li>Normalize ข้อมูลตัวเลขด้วย MinMaxScaler ให้อยู่ในช่วง 0-1</li>
    </ul>
    </div>

    <div style='text-align: left; margin-left: 60px;'>
        <h2 style='font-size: 40px;'>ทฤษฎีของอัลกอริทึม Neural Network</h2>
    </div>
    <div style='font-size:20px; line-height: 1.6; margin-left: 0px;'>
    <h3 style='font-size: 24px;'>การนำมาใช้กับ Dataset นี้</h3>
    <p>Dataset นี้เกี่ยวข้องกับข้อมูลกิจกรรมในตอนเช้าและการเดินทาง เช่น เวลาอาบน้ำ เตรียมตัว และการเดินทางหลายต่อ เพื่อทำนายว่า "ควรตื่นกี่โมง" ดังนั้น Neural Network สามารถเรียนรู้จากความสัมพันธ์ที่ซับซ้อน เช่น จำนวนพาหนะ ระยะเวลาเดินทาง และเวลาที่ต้องเผื่อไว้ เพื่อหาค่าที่เหมาะสมที่สุดของเวลาตื่น</p>

    <h3 style='font-size: 24px;'>Feedforward Neural Network</h3>
    <ul>
    <li>ประกอบด้วยหลายเลเยอร์: Input Layer → Hidden Layers → Output Layer</li>
    <li>ข้อมูลจะไหลจาก Input ไปยัง Output โดยไม่มีการย้อนกลับ</li>
    <li>แต่ละโหนดจะคำนวณค่าจากการถ่วงน้ำหนักและรวมค่าจากเลเยอร์ก่อนหน้า</li>
    <li>ใช้ Activation Function เช่น ReLU สำหรับ Hidden Layers และ Linear สำหรับ Output</li>
    <li>Optimizer: Adam ช่วยปรับพารามิเตอร์เพื่อให้ค่าคลาดเคลื่อนลดลง</li>
    <li>Loss Function: Mean Squared Error (MSE) ใช้วัดความผิดพลาดของผลลัพธ์</li>
    </ul>
    <p>จุดเด่นของ Neural Network คือสามารถจับความสัมพันธ์ที่ไม่เป็นเชิงเส้นของข้อมูลได้ดี เหมาะสำหรับปัญหาที่ซับซ้อน เช่น การทำนายเวลาตื่นจากกิจกรรมและการเดินทางที่หลากหลาย</p>
    </div>

    <div style='text-align: left; margin-top: 20px; margin-left: 0px;'>
        <h2 style='font-size: 40px;'>ขั้นตอนการพัฒนาโมเดล Neural Network</h2>
    </div>
    <div style='font-size:20px; line-height:  1.6; margin-left: 0px;'>
    <ol>
    <li>เตรียม Dataset และทำ Data Cleaning + Normalize</li>
    <li>แบ่งข้อมูลเป็น Training Set (80%) และ Test Set (20%)</li>
    <li>สร้างโมเดล Neural Network ด้วย TensorFlow/Keras</li>
    <li>เทรนโมเดลด้วย Early Stopping เพื่อป้องกัน Overfitting</li>
    <li>ประเมินผลด้วย MAE (Mean Absolute Error) และ Loss</li>
    </ol>
    </div>

    <div style='text-align: left; margin-top: 30px;'>
        <h2 style='font-size: 36px;'>ฟีเจอร์ใน Dataset</h2>
    </div>
    <div style='font-size:20px; text-align: left; line-height: 1.6;'>
    <ul>
    <li><b>Class_Start</b>: เวลาที่ต้องถึงห้องเรียน (HH:MM)</li>
    <li><b>WakeUp_Time</b>: เวลาที่ควรตื่น (HH:MM) - ค่าที่ Neural Network จะทำนาย</li>
    <li><b>Start_Shower / Start_DressUp / Start_Prepare_Items</b>: เวลาที่เริ่มทำกิจกรรมต่าง ๆ (HH:MM)</li>
    <li><b>Shower_Duration / DressUp_Duration / Prepare_Items_Duration</b>: ระยะเวลาในการอาบน้ำ, แต่งตัว, เตรียมของ (นาที)</li>
    <li><b>Transport_Mode_1 / 2 / 3</b>: ประเภทพาหนะในการเดินทางแต่ละต่อ เช่น Walk, Bus, MRT (เข้ารหัสด้วย Label Encoding)</li>
    <li><b>Time_1 / 2 / 3</b>: เวลาที่ใช้เดินทางแต่ละต่อ (นาที)</li>
    <li><b>Wait_1 / 2 / 3</b>: เวลาที่ต้องรอพาหนะในแต่ละต่อ (นาที)</li>
    <li><b>Total_Commute_Time</b>: เวลารวมในการเดินทาง (นาที)</li>
    <li><b>Leave_Home</b>: เวลาที่ต้องออกจากบ้าน (HH:MM)</li>
    <li><b>Arrive_Class</b>: เวลาที่ไปถึงห้องเรียน (HH:MM)</li>
    <li><b>Buffer_Time</b>: เวลาสำรองเผื่อ (นาที)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div style='text-align: left; margin-top: 50px;'><h2 style='font-size: 40px;'>Dataset ที่ใช้</h2></div>", unsafe_allow_html=True)

    file_id = "1JJ1nA6eFYv3Xq7R09EOLz4K8CUN9Rwsr"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        response = requests.get(url)
        with open("WakeUpSchedulerDataset.csv", "wb") as f:
            f.write(response.content)

        df = pd.read_csv("WakeUpSchedulerDataset.csv")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 ดาวน์โหลด Dataset", data=csv, file_name="WakeUpSchedulerDataset.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ โหลด Dataset ไม่สำเร็จ: {e}")
