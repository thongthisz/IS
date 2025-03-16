import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Machine Demo", layout="wide", initial_sidebar_state="collapsed")

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
        <a class='nav-link' href="/Data_Preparation.py"> Machine Info</a>
        <a class='nav-link' href="/ML.py"> Machine Test</a>
        <a class='nav-link' href="/Model_Explanation.py"> NN Info</a>
        <a class='nav-link' href="/NN.py"> NN Test</a>
    </div>
""", unsafe_allow_html=True)

st.title("ทำนายราคารถมอเตอร์ไซค์ด้วย Machine Learning")

with open("linear_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# โหลด Dataset
df = pd.read_csv("data/MotorcycleDataset.csv")
df = df.dropna()

st.header("📄 เลือกข้อมูลเพื่อทำนายราคา")

brands_all = sorted(df['Brand'].unique())
types_all = sorted(df['Type'].unique())
provinces_all = sorted(df['Province'].unique())
conditions = sorted(df['Condition'].unique())

for key in ['brand', 'type', 'province', 'model', 'cc']:
    if key not in st.session_state:
        st.session_state[key] = "-- กรุณาเลือก --"

filtered_df = df.copy()
if st.session_state['brand'] != "-- กรุณาเลือก --":
    filtered_df = filtered_df[filtered_df['Brand'] == st.session_state['brand']]
if st.session_state['type'] != "-- กรุณาเลือก --":
    filtered_df = filtered_df[filtered_df['Type'] == st.session_state['type']]
if st.session_state['province'] != "-- กรุณาเลือก --":
    filtered_df = filtered_df[filtered_df['Province'] == st.session_state['province']]

models_filtered = sorted(filtered_df['Model'].unique())

def update_brand():
    st.session_state['brand'] = st.session_state.brand_select

def update_type():
    st.session_state['type'] = st.session_state.type_select

def update_province():
    st.session_state['province'] = st.session_state.province_select

def update_model():
    st.session_state['model'] = st.session_state.model_select
    if st.session_state['model'] != "-- กรุณาเลือก --":
        temp_df = filtered_df[filtered_df['Model'] == st.session_state['model']]
        if not temp_df.empty:
            st.session_state['cc'] = int(temp_df['CC'].mode()[0])

col1, col2 = st.columns(2)
col1.selectbox("ยี่ห้อ (Brand)", ["-- กรุณาเลือก --"] + brands_all, key="brand_select", index=0 if st.session_state['brand'] == "-- กรุณาเลือก --" else brands_all.index(st.session_state['brand'])+1, on_change=update_brand)
col2.selectbox("ประเภทรถ (Type)", ["-- กรุณาเลือก --"] + types_all, key="type_select", index=0 if st.session_state['type'] == "-- กรุณาเลือก --" else types_all.index(st.session_state['type'])+1, on_change=update_type)
col1.selectbox("จังหวัด (Province)", ["-- กรุณาเลือก --"] + provinces_all, key="province_select", index=0 if st.session_state['province'] == "-- กรุณาเลือก --" else provinces_all.index(st.session_state['province'])+1, on_change=update_province)
col2.selectbox("รุ่น (Model)", ["-- กรุณาเลือก --"] + models_filtered, key="model_select", index=0 if st.session_state['model'] == "-- กรุณาเลือก --" else (models_filtered.index(st.session_state['model'])+1 if st.session_state['model'] in models_filtered else 0), on_change=update_model)

if st.session_state['cc'] != "-- กรุณาเลือก --":
    col1.markdown(f"**CC:** {st.session_state['cc']}")

distance = col2.number_input("ระยะทาง (กม)", min_value=0, max_value=120000, value=0)
selected_condition = col1.selectbox("สภาพรถ (Condition)", conditions)
age = col2.number_input("อายุของรถ (ปี)", min_value=0, max_value=30, value=0)

if st.button("ทำนายราคา"):
    if "-- กรุณาเลือก --" in [st.session_state['brand'], st.session_state['model'], st.session_state['type'], st.session_state['province']] or st.session_state['cc'] == "-- กรุณาเลือก --":
        st.error("กรุณาเลือกข้อมูลให้ครบทุกช่องก่อนทำนาย")
    else:
        input_data = pd.DataFrame([[
            encoders['Brand'].transform([st.session_state['brand']])[0],
            encoders['Model'].transform([st.session_state['model']])[0],
            st.session_state['cc'],
            encoders['Type'].transform([st.session_state['type']])[0],
            encoders['Province'].transform([st.session_state['province']])[0],
            distance,
            encoders['Condition'].transform([selected_condition])[0],
            age
        ]], columns=['Brand', 'Model', 'CC', 'Type', 'Province', 'Distance', 'Condition', 'Age'])

        lr_price = lr_model.predict(input_data)[0]
        rf_price = rf_model.predict(input_data)[0]
        avg_price = round((lr_price + rf_price) / 2, -2)

        st.subheader("📈 ราคาประเมิน")
        st.write(f"### 💰 ราคาประเมินโดยรวม: **{int(avg_price):,} บาท**")

        with st.expander("🔍 ดูรายละเอียดการทำนาย"):
            st.write(f"Linear Regression ทำนาย: **{int(round(lr_price, -2)):,} บาท**")
            st.write(f"Random Forest ทำนาย: **{int(round(rf_price, -2)):,} บาท**")
