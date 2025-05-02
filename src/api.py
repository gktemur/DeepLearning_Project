from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from src.churn.feature_engineering import FeatureEngineering, FeatureConfig
from src.churn.model import ChurnPredictor
import joblib
from src.iade.feature_engineering import IadeFeatureEngineering, IadeFeatureConfig
from src.iade.model import IadeRiskPredictor
from src.product.model import ProductPurchasePredictor
from src.product.feature_engineering import ProductFeatureEngineering
import tensorflow as tf

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
scaler = None

# Initialize iade risk predictor
iade_feature_engineering = IadeFeatureEngineering()
iade_model = None
iade_scaler = None

# Load model and feature engineering
model_product = None
feature_engineering_product = None

class CustomerData(BaseModel):
    """Input data model for customer features"""
    total_order_value: float
    order_count: int
    average_order_value: float
    recency_days: int
    frequency_score: int
    monetary_score: float
    is_recent: int
    is_one_time_customer: int
    is_high_value: int
    is_frequent: int
    is_low_spender: int
    has_large_order: int
    recency_score: float

class PredictionResponse(BaseModel):
    """Output data model for predictions"""
    churn_probability: float
    will_churn: bool
    confidence: float

class IadeData(BaseModel):
    """Input data model for return risk prediction"""
    discount: float
    quantity: int
    total_spending: float

class IadePredictionResponse(BaseModel):
    """Output data model for return risk predictions"""
    risk_probability: float
    is_risky: bool
    confidence: float

class CustomerFeatures(BaseModel):
    """Customer features for prediction"""
    customer_id: str
    category_spending: Dict[str, float]  # Dictionary of category:spending pairs

class ProductPrediction(BaseModel):
    product_name: str
    purchase_probability: float
    recommendation: bool

class PredictionResponseProduct(BaseModel):
    """Response model for predictions"""
    customer_id: str
    predictions: List[ProductPrediction]
    recommended_products: List[str]

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    try:
        if model is None:
            # Load and prepare data for model initialization
            X, y = feature_engineering.prepare_data()
            model = ChurnPredictor(input_dim=X.shape[1])
            
            # Try to load saved model weights
            model_path = os.path.join('models', 'churn_model.h5')
            if os.path.exists(model_path):
                model.load_model(model_path)
                print(f"Model loaded from {model_path}")
            else:
                print("No saved model found. Using untrained model.")
            
            # Save scaler for later use
            scaler = feature_engineering.scaler
            joblib.dump(scaler, 'models/scaler.joblib')
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def load_iade_model():
    """Load the trained iade model and scaler"""
    global iade_model, iade_scaler
    try:
        if iade_model is None:
            # Load and prepare data for model initialization
            X_train, _, y_train, _ = iade_feature_engineering.prepare_data()
            iade_model = IadeRiskPredictor(input_dim=X_train.shape[1])
            
            # Try to load saved model
            model_path = os.path.join('models', 'best_iade_model.keras')
            if os.path.exists(model_path):
                iade_model.load_model(model_path)
                print(f"Iade model loaded from {model_path}")
            else:
                print("No saved iade model found. Using untrained model.")
            
            # Load scaler
            scaler_path = os.path.join('models', 'iade_scaler.joblib')
            if os.path.exists(scaler_path):
                iade_scaler = joblib.load(scaler_path)
                print(f"Iade scaler loaded from {scaler_path}")
    except Exception as e:
        print(f"Error loading iade model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_model()
    load_iade_model()
    
    try:
        # Initialize feature engineering
        global feature_engineering_product, model_product
        feature_engineering_product = ProductFeatureEngineering()
        
        # Initialize model with correct dimensions
        model_product = ProductPurchasePredictor(input_dim=8)  # 8 categories
        
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
                feature_engineering_product.scaler = joblib.load(scaler_path)
                print(f"Scaler loaded from {scaler_path}")
            except Exception as e:
                print(f"Error loading scaler from {scaler_path}: {str(e)}")
        
    except Exception as e:
        print(f"Error initializing product model: {str(e)}")
        # Initialize empty model
        model_product = ProductPurchasePredictor(input_dim=8)
        model_product.model = model_product._build_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "endpoints": {
            "/predict": "POST - Make churn predictions",
            "/health": "GET - Check API health",
            "/predict-iade": "POST - Predict return risk for an order",
            "/predict-product": "POST - Predict product purchase probabilities"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model_product.model is not None,
        "scaler_loaded": feature_engineering_product.scaler is not None
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
            customer_data.monetary_score,
            customer_data.is_recent,
            customer_data.is_one_time_customer,
            customer_data.is_high_value,
            customer_data.is_frequent,
            customer_data.is_low_spender,
            customer_data.has_large_order,
            customer_data.recency_score
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        churn_probability = float(model.predict(features_scaled)[0][0])
        will_churn = churn_probability > 0.4
        confidence = abs(churn_probability - 0.4) * 2
        
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
            customer.monetary_score,
            customer.is_recent,
            customer.is_one_time_customer,
            customer.is_high_value,
            customer.is_frequent,
            customer.is_low_spender,
            customer.has_large_order,
            customer.recency_score
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

@app.post("/predict-iade", response_model=IadePredictionResponse)
async def predict_iade_risk(iade_data: IadeData):
    """
    Predict return risk for an order
    
    Args:
        iade_data: Order features
        
    Returns:
        Prediction results including probability and confidence
    """
    try:
        # Convert input data to numpy array
        features = np.array([[
            iade_data.discount,
            iade_data.quantity,
            iade_data.total_spending
        ]])
        
        # Scale features
        features_scaled = iade_scaler.transform(features)
        
        # Make prediction
        risk_probability = float(iade_model.predict(features_scaled)[0][0])
        is_risky = risk_probability > 0.5
        confidence = abs(risk_probability - 0.5) * 2
        
        return IadePredictionResponse(
            risk_probability=risk_probability,
            is_risky=is_risky,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-product", response_model=PredictionResponseProduct)
async def predict_product_purchase(customer_features: CustomerFeatures):
    """Predict purchase probabilities for new products"""
    if model_product.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Prepare customer features
        features = feature_engineering_product.prepare_customer_features(
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
        
        return PredictionResponseProduct(
            customer_id=customer_features.customer_id,
            predictions=product_predictions,
            recommended_products=recommended_products
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get information about the model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "input_dimension": model.input_dim,
        "embedding_dimension": model.embedding_dim,
        "number_of_products": model.n_products,
        "model_architecture": model.model.summary()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 