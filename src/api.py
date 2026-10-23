import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MLSecOps Threat Detection API")

# Load the trained model and scaler on startup
model = joblib.load('models/threat_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Define the expected JSON payload structure
class NetworkData(BaseModel):
    features: list[float]

@app.get("/")
def health_check():
    return {"status": "active", "model": "Random Forest NSL-KDD"}

@app.post("/predict")
def predict_threat(data: NetworkData):
    # The model expects a 2D array
    prediction = model.predict([data.features])
    result = "Attack" if prediction[0] == 1 else "Normal"
    return {"prediction": result}