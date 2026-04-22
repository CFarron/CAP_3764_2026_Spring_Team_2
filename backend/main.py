# Import FastAPI (main web framework) & HTTPException(return errors)
from fastapi import FastAPI, HTTPException
# Import HTMLResponse (return HTML from endpoints) and JSONResponse (return JSON from endpoints)
from fastapi.responses import HTMLResponse, JSONResponse
# Import BaseModel (defines data schemas for request bodies)
from pydantic import BaseModel
# Import List (type hint for lists of item)
from typing import List
# Load the trained model
import joblib

from pathlib import Path

import numpy as np
import pandas as pd


# Run the app: 
    # 1-Open a terminal
    # 2-Activate env (run in terminal: conda activate fastapi_streamlit)
    # 3-Set your working directory to the folder where you have main.py (run in terminal: cd your_path/)
    # 4-Execute main.py (run in terminal: uvicorn main:app --reload --port 8000)



# -------------------------------------------------
# 1- Configuration
# -------------------------------------------------
# These MUST match the DataFrame column names used to train model.pkl
FEATURES = ["month", "day_of_week", "dep_hour", "distance", "is_peak", "is_weekend", "is_early_morning"]

# -------------------------------------------------
# 2-Load trained model
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "xgb_delay_model.pkl"
model = joblib.load(model_path)
print(f"Model loaded successfully from {model_path}")

# Initialize the FastAPI app by creating the web API server and defining metadata that appears in /docs.
app = FastAPI(
    title="MIA Flight Delay Prediction API",
    description="Predicts the probability that a flight will be delayed"
)

# -------------------------------------------------
# 3-Pydantic models (Define the data your API expects and returns)
# -------------------------------------------------
# Input schema for a single observation (defines JSON fields needed to make predictions, which FastAPI uses to validate incoming requests)
class InputData(BaseModel):
    month: int
    day_of_week: int
    dep_hour: int
    distance: int

class PredictionOut(BaseModel):
    probability: float

# Output schema for each prediction in batch mode (each prediction will include the position in the input list and the predicted probability)
class BatchPredictionItem(BaseModel):
    index: int
    probability: float

# Output schema for the whole batch (defines the shape of the full response for /batch_predict)
class BatchPredictionOut(BaseModel):
    predictions: List[BatchPredictionItem]

# -------------------------------------------------
# 4 - Feature engineering
# -------------------------------------------------
def build_features(data: InputData) -> pd.DataFrame:
    is_peak = 1 if data.dep_hour in [6, 7, 8, 16, 17, 18] else 0
    is_weekend = 1 if data.day_of_week in [6, 7] else 0
    is_early_morning = 1 if data.dep_hour < 8 else 0

    row = {
        "month": data.month,
        "day_of_week": data.day_of_week,
        "dep_hour": data.dep_hour,
        "distance": data.distance,
        "is_peak": is_peak,
        "is_weekend": is_weekend,
        "is_early_morning": is_early_morning,
    }

    return pd.DataFrame([row], columns=FEATURES)


def build_batch_features(items: List[InputData]) -> pd.DataFrame:
    records = []

    for item in items:
        is_peak = 1 if item.dep_hour in [6, 7, 8, 16, 17, 18] else 0
        is_weekend = 1 if item.day_of_week in [6, 7] else 0
        is_early_morning = 1 if item.dep_hour < 8 else 0

        records.append({
            "month": item.month,
            "day_of_week": item.day_of_week,
            "dep_hour": item.dep_hour,
            "distance": item.distance,
            "is_peak": is_peak,
            "is_weekend": is_weekend,
            "is_early_morning": is_early_morning,
        })

    return pd.DataFrame(records, columns=FEATURES)


# -------------------------------------------------
# 5-Endpoints
# -------------------------------------------------
# a-Root endpoint (returns a simple HTML welcome page)
@app.get("/")
async def main():
    content = """
    <body>
    <h1>Welcome to the MIA Flight Delay Prediction</h1>
    <p>This API predicts the probability of a flight delay</p>
    <p>Navigate to <code>/docs</code> to see the API documentation.</p>
    <body>
    """
    return HTMLResponse(content)


# b-Health check (useful for monitoring and debugging deployment)
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }


# c-Single prediction endpoint
@app.post("/predict")
async def predict(data: InputData):
    """Single-row prediction."""
    try:
        X = build_features(data)
        prob = model.predict_proba(X)[0, 1]
        return PredictionOut(probability=float(prob))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


# d-Batch prediction endpoint
@app.post("/batch_predict", response_model=BatchPredictionOut)
async def batch_predict(items: List[InputData]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not items:
        raise HTTPException(status_code=400, detail="Empty input list")

    try:
        X = build_batch_features(items)
        probs = model.predict_proba(X)[:, 1]

        predictions = [
            BatchPredictionItem(index=i, probability=float(p))
            for i, p in enumerate(probs)
        ]

        return BatchPredictionOut(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")
