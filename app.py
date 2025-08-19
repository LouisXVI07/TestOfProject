from fastapi import FastAPI
import joblib
import pandas as pd

# Load model and encoder
model = joblib.load("best_crop_model.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Crop Prediction API is running 🚀"}

@app.post("/predict")
def predict_crop(data: dict):
    # Convert input into DataFrame
    df = pd.DataFrame([data])
    
    # Predict encoded label
    pred_encoded = model.predict(df)[0]
    
    # Convert to crop name
    pred_label = label_encoder.inverse_transform([int(pred_encoded)])[0]
    
    return {"Predicted Crop": pred_label}
