from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import tensorflow as tf
from src.churn import ChurnPredictor, ChurnFeatureEngineering
import joblib
import os

router = APIRouter()

# Initialize feature engineering and model
feature_engineering = ChurnFeatureEngineering()
model_churn = ChurnPredictor()

# Load model if exists
model_paths = [
    'models/best_churn_model.keras',
    'models/best_model.keras',
    'models/churn_model.keras'
]

for path in model_paths:
    if os.path.exists(path):
        try:
            model_churn.model = tf.keras.models.load_model(path)
            print(f"Model loaded from {path}")
            break
        except Exception as e:
            print(f"Error loading model from {path}: {str(e)}")

# Load scaler if exists
scaler_path = 'models/churn_scaler.joblib'
if os.path.exists(scaler_path):
    try:
        feature_engineering.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {str(e)}")

class CustomerFeatures(BaseModel):
    customer_id: str
    features: Dict[str, float]

class ChurnPrediction(BaseModel):
    churn_probability: float
    prediction: bool
    risk_level: str

@router.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer_features: CustomerFeatures):
    """Predict customer churn probability"""
    if model_churn.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Prepare customer features
        features = feature_engineering.prepare_customer_features(
            customer_features.features
        )
        
        # Make prediction
        prediction = model_churn.predict(features)
        prob = float(prediction[0][0])
        
        # Determine risk level
        if prob < 0.3:
            risk_level = "Low"
        elif prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return ChurnPrediction(
            churn_probability=prob,
            prediction=prob > 0.5,
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

def is_healthy():
    """Check if the churn prediction service is healthy"""
    return {
        "model_loaded": model_churn.model is not None,
        "scaler_loaded": feature_engineering.scaler is not None
    } 