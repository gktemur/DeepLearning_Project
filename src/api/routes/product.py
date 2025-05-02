from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import tensorflow as tf
from src.product import ProductPurchasePredictor, ProductFeatureEngineering
import joblib
import os

router = APIRouter()

# Initialize feature engineering and model
feature_engineering = ProductFeatureEngineering()
model_product = ProductPurchasePredictor()

# Load model if exists
model_paths = [
    'models/best_product_model.keras',
    'models/best_model.keras',
    'models/product_model.keras'
]

for path in model_paths:
    if os.path.exists(path):
        try:
            model_product.model = tf.keras.models.load_model(path)
            print(f"Model loaded from {path}")
            break
        except Exception as e:
            print(f"Error loading model from {path}: {str(e)}")

# Load scaler if exists
scaler_path = 'models/product_scaler.joblib'
if os.path.exists(scaler_path):
    try:
        feature_engineering.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {str(e)}")

class CustomerFeatures(BaseModel):
    customer_id: str
    category_spending: Dict[str, float]

class ProductPrediction(BaseModel):
    product_name: str
    purchase_probability: float
    recommendation: bool

class PredictionResponse(BaseModel):
    customer_id: str
    predictions: List[ProductPrediction]
    recommended_products: List[str]

@router.post("/predict", response_model=PredictionResponse)
async def predict_product_purchase(customer_features: CustomerFeatures):
    """Predict purchase probabilities for new products"""
    if model_product.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Prepare customer features
        features = feature_engineering.prepare_customer_features(
            customer_features.category_spending
        )
        
        # Make predictions
        predictions = model_product.predict(features)
        
        # Format response
        product_names = ["SmartWatch_Pro", "SportRunner_X", "KitchenMaster_AI"]
        product_predictions = []
        recommended_products = []
        
        for i, (product_name, prob) in enumerate(zip(product_names, predictions[0])):
            product_predictions.append(
                ProductPrediction(
                    product_name=product_name,
                    purchase_probability=float(prob),
                    recommendation=prob > 0.5
                )
            )
            if prob > 0.5:
                recommended_products.append(product_name)
        
        return PredictionResponse(
            customer_id=customer_features.customer_id,
            predictions=product_predictions,
            recommended_products=recommended_products
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

def is_healthy():
    """Check if the product prediction service is healthy"""
    return {
        "model_loaded": model_product.model is not None,
        "scaler_loaded": feature_engineering.scaler is not None
    } 