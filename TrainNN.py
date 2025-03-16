import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle

print("🚀 เริ่มโหลด Dataset ...")
file_path = "data/WakeUpSchedulerDataset.csv"
df = pd.read_csv(file_path)

# ลบ missing values
df_cleaned = df.dropna()

# แปลงเวลาเป็นนาที
for col in ['Class_Start', 'WakeUp_Time', 'Start_Shower', 'Start_DressUp', 'Start_Prepare_Items', 'Leave_Home', 'Arrive_Class']:
    df_cleaned[col] = pd.to_datetime(df_cleaned[col], format="%H:%M").dt.hour * 60 + pd.to_datetime(df_cleaned[col], format="%H:%M").dt.minute

# เข้ารหัสพาหนะ
for col in ['Transport_Mode_1', 'Transport_Mode_2', 'Transport_Mode_3']:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])

# เลือก Features
features = ['Shower_Duration', 'DressUp_Duration', 'Prepare_Items_Duration',
            'Transport_Mode_1', 'Time_1', 'Wait_1',
            'Transport_Mode_2', 'Time_2', 'Wait_2',
            'Transport_Mode_3', 'Time_3', 'Wait_3',
            'Total_Commute_Time', 'Buffer_Time']
target = 'WakeUp_Time'

X = df_cleaned[features]
y = df_cleaned[target]

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# เตรียมโฟลเดอร์สำหรับบันทึก
model_dir = "Trained_data"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "wake_up_model.h5")
scaler_path = os.path.join(model_dir, "scaler.pkl")

# สร้างและเทรนโมเดล
print("⚙️ เริ่มเทรนโมเดล ...")
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

# บันทึกโมเดลและ scaler
model.save(model_path)
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

# ประเมินผล
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ โมเดลถูกบันทึกที่: {model_path}")
print(f"✅ Scaler ถูกบันทึกที่: {scaler_path}")
print(f"🎯 MAE: {mae:.2f} นาที")
