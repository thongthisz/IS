import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# โหลด Dataset
df = pd.read_csv("data/MotorcycleDataset.csv")

# ลบ Missing Values
df = df.dropna()

# Label Encoding สำหรับคอลัมน์หมวดหมู่
label_cols = ['Brand', 'Model', 'Type', 'Province', 'Condition']
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features และ Target
X = df[['Brand', 'Model', 'CC', 'Type', 'Province', 'Distance', 'Condition', 'Age']]
y = df['Price']

# แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# เทรน Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# เทรน Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# เซฟโมเดล
with open("linear_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# เซฟ encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("✅ โมเดลและ Encoders ถูกบันทึกแล้วเรียบร้อย!")
