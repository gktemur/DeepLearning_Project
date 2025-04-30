from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import joblib
import os
from dotenv import load_dotenv
from feature_engineering import FeatureEngineering, FeatureConfig
from model import ChurnPredictor
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using deep learning",
    version="1.0.0"
)

# Initialize feature engineering and model
feature_engineering = FeatureEngineering(FeatureConfig())
model = None
scaler = StandardScaler()

class CustomerData(BaseModel):
    """Input data model for customer features"""
    total_order_value: float
    order_count: int
    average_order_value: float
    recency_days: int
    frequency_score: int
    monetary_score: float

class PredictionResponse(BaseModel):
    """Output data model for predictions"""
    churn_probability: float
    will_churn: bool
    confidence: float

def load_model():
    """Load the trained model"""
    global model, scaler
    try:
        if model is None:
            # Load and prepare data for model initialization
            X, y = feature_engineering.prepare_data()
            model = ChurnPredictor(input_dim=X.shape[1])
            
            # Fit scaler with training data
            scaler.fit(X)
            
            # Try to load saved model weights
            model_path = os.path.join('models', 'churn_model.h5')
            if os.path.exists(model_path):
                model.load_model(model_path)
                print(f"Model loaded from {model_path}")
            else:
                print("No saved model found. Using untrained model.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "endpoints": {
            "/predict": "POST - Make churn predictions",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """
    Predict customer churn
    
    Args:
        customer_data: Customer features
        
    Returns:
        Prediction results including probability and confidence
    """
    try:
        # Convert input data to numpy array
        features = np.array([[
            customer_data.total_order_value,
            customer_data.order_count,
            customer_data.average_order_value,
            customer_data.recency_days,
            customer_data.frequency_score,
            customer_data.monetary_score
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        churn_probability = float(model.predict(features_scaled)[0][0])
        will_churn = churn_probability > 0.5
        confidence = abs(churn_probability - 0.5) * 2  # Convert to 0-1 scale
        
        return PredictionResponse(
            churn_probability=churn_probability,
            will_churn=will_churn,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch", response_model=List[PredictionResponse])
async def predict_churn_batch(customers_data: List[CustomerData]):
    """
    Predict churn for multiple customers
    
    Args:
        customers_data: List of customer features
        
    Returns:
        List of prediction results
    """
    try:
        # Convert input data to numpy array
        features = np.array([[
            customer.total_order_value,
            customer.order_count,
            customer.average_order_value,
            customer.recency_days,
            customer.frequency_score,
            customer.monetary_score
        ] for customer in customers_data])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        probabilities = model.predict(features_scaled)
        
        # Prepare response
        predictions = []
        for prob in probabilities:
            churn_probability = float(prob[0])
            will_churn = churn_probability > 0.5
            confidence = abs(churn_probability - 0.5) * 2
            
            predictions.append(PredictionResponse(
                churn_probability=churn_probability,
                will_churn=will_churn,
                confidence=confidence
            ))
        
        return predictions
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 