import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Machine Info", layout="wide", initial_sidebar_state="collapsed")

# แสดง Navbar
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

# แบ่ง Column
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div style='margin-top: 50px; margin-left: 100px;'>
        <h2 style='font-size: 40px;'>การเตรียมข้อมูล (Data Preparation)</h2>
    </div>
    <div style='font-size:20px; text-align: left; line-height: 1.6;'>
    <ul>
    <li>ตรวจสอบข้อมูลที่ขาดหาย (Missing Values)</li>
    <li>ลบแถวที่มีข้อมูลขาด หรือเติมค่ากลาง เช่น ค่าเฉลี่ย (Mean), ค่าที่พบบ่อย (Mode)</li>
    <li>แปลงข้อมูลหมวดหมู่ เช่น Brand, Type, Condition ด้วย One-Hot Encoding หรือ Label Encoding</li>
    <li>Normalize ข้อมูลตัวเลข เช่น CC, Distance, Price ให้อยู่ในช่วง 0-1</li>
    <li>ตรวจสอบและจัดการ Outliers เช่น ราคาสูงผิดปกติ หรือ CC เกินจริง</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-left: 60px;'>
        <h2 style='font-size: 40px;'>ทฤษฎีของอัลกอริทึม Machine Learning</h2>
    </div>
    <div style='font-size:20px; line-height: 1.6;'>
    <h3 style='font-size: 24px;'>Linear Regression</h3>
    <ul>
    <li>โมเดลพื้นฐานที่ใช้สมการเส้นตรง \( y = wx + b \)</li>
    <li>หาความสัมพันธ์ระหว่างฟีเจอร์ต่าง ๆ เช่น CC, Age, Distance กับราคาของรถ</li>
    <li>จุดเด่น: เข้าใจง่าย รันเร็ว เหมาะสำหรับข้อมูลเชิงเส้น</li>
    </ul>

    <h3 style='font-size: 24px;'>Random Forest Regression</h3>
    <ul>
    <li>โมเดลแบบ Ensemble ที่รวมผลจากหลาย Decision Trees</li>
    <li>สามารถจัดการกับข้อมูลที่ซับซ้อนได้ดี และลดปัญหา Overfitting</li>
    <li>จุดเด่น: แม่นยำกว่าการใช้ Decision Tree เพียงต้นเดียว รองรับข้อมูลที่ไม่เป็นเชิงเส้น</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 20px;'>
        <h2 style='font-size: 40px;'>ขั้นตอนการพัฒนาโมเดล Machine Learning</h2>
    </div>
    <div style='font-size:20px; line-height: 1.6;'>
    <ol>
    <li>แบ่งข้อมูลเป็น Training Set (80%) และ Test Set (20%)</li>
    <li>เทรนโมเดลด้วย Linear Regression และ Random Forest Regression</li>
    <li>ประเมินผลโมเดลด้วยค่า MAE (Mean Absolute Error) และ RMSE</li>
    <li>เปรียบเทียบผลลัพธ์ของทั้งสองโมเดล เพื่อเลือกโมเดลที่ดีที่สุดสำหรับการทำนายราคา</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 30px;'>
        <h2 style='font-size: 40px;'>รายละเอียดฟีเจอร์ใน Dataset</h2>
    </div>
    <div style='font-size:20px; line-height: 1.6;'>
    <ul>
    <li><b>Brand</b>: ยี่ห้อของรถ เช่น Honda, Yamaha, Vespa</li>
    <li><b>Model</b>: รุ่นของรถ เช่น PCX 160, Click 125</li>
    <li><b>CC</b>: ขนาดเครื่องยนต์ (ซีซี)</li>
    <li><b>Type</b>: ประเภทรถ เช่น Underbone, Scooter, Sport</li>
    <li><b>Province</b>: จังหวัดที่รถอยู่</li>
    <li><b>Distance</b>: ระยะทางที่รถใช้งานมาแล้ว (กิโลเมตร)</li>
    <li><b>Condition</b>: สภาพรถ เช่น Excellent, Good, Fair, Poor, Broken</li>
    <li><b>Age</b>: อายุของรถ (ปี)</li>
    <li><b>Price</b>: ราคาที่ใช้ในการทำนาย (บาท) – Target ที่ต้องการหา</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div style='margin-top: 50px;'><h2 style='font-size: 40px;'>Dataset ที่ใช้</h2></div>", unsafe_allow_html=True)

    # ลองเปลี่ยน path หากไม่เจอไฟล์
    file_path1 = os.path.join("pages", "MotorcycleDataset.csv")
    if os.path.exists(file_path1):
        df1 = pd.read_csv(file_path1)
        st.dataframe(df1, use_container_width=True)

        # ปุ่มโหลดไฟล์
        csv = df1.to_csv(index=False).encode('utf-8')
        st.download_button("ดาวน์โหลด Dataset", data=csv, file_name='MotorcycleDataset.csv', mime='text/csv')
    else:
        st.error("❌ ไม่พบไฟล์ MotorcycleDataset.csv กรุณาตรวจสอบตำแหน่งไฟล์")

