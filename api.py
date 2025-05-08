from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Prediksi Harga Rumah API")

# Muat model dan scaler
model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class PredictionInput(BaseModel):
    luas_meter2: float
    kamar_tidur: int
    kamar_mandi: int
    jarak_ke_kota: float
    tahun_bangun: int

@app.post("/predict")
async def predict_price(input: PredictionInput):
    input_data = np.array([[
        input.luas_meter2,
        input.kamar_tidur,
        input.kamar_mandi,
        input.jarak_ke_kota,
        input.tahun_bangun
    ]])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    
    return {"harga_prediksi": float(prediction)}