import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Pastikan direktori model ada
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv("data/cleaned_dataset.csv")

# Features & Target
X = df[['luas_meter2', 'kamar_tidur', 'kamar_mandi', 'jarak_ke_kota', 'tahun_bangun']]
y = df['harga']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model dan scaler
joblib.dump(model, "models/house_price_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model dan scaler berhasil disimpan di folder 'models/'.")