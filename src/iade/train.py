import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd
from src.iade.feature_engineering import IadeFeatureEngineering, IadeFeatureConfig
from src.iade.model import IadeRiskPredictor
import joblib


def train_iade_model():
    """Train the return risk prediction model"""
    print("Starting return risk prediction model training...")
    
    # Initialize feature engineering
    config = IadeFeatureConfig()
    feature_engineering = IadeFeatureEngineering(config)
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = feature_engineering.prepare_data()
    
    # Further split training data into train and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = IadeRiskPredictor(input_dim=X_train.shape[1])
    
    # Train model with cost-sensitive learning
    print("\nTraining model...")
    model.train(
        X_train=X_train.to_numpy(),
        y_train=y_train.to_numpy(),
        X_val=X_val.to_numpy(),
        y_val=y_val.to_numpy(),
        epochs=100,
        batch_size=32,
        cost_ratio=5.0  # Higher cost for missing risky orders
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test.to_numpy(), y_test.to_numpy())
    print("\nModel Evaluation Metrics:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save model
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/iade_model.keras')
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_iade_model() 