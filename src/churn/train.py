import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd
from src.churn.feature_engineering import FeatureEngineering, FeatureConfig
from src.churn.model import ChurnPredictor, prepare_data_for_modeling
import joblib


def train_churn_model():
    """Train the churn prediction model"""
    print("Starting churn prediction model training...")
    
    # Initialize feature engineering
    config = FeatureConfig()
    feature_engineering = FeatureEngineering(config)
    
    # Prepare data
    print("\nPreparing data...")
    X, y = feature_engineering.prepare_data()
    
    # Split data
    print("\nSplitting data into train/val/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_modeling(X, y)
    
    # Initialize model
    print("\nInitializing model...")
    model = ChurnPredictor(input_dim=X.shape[1])
    
    # Train model
    print("\nTraining model...")
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Evaluation Metrics:")
    print(metrics['classification_report'])
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/churn_model.weights.h5')
    
    # Plot results
    print("\nGenerating plots...")
    model.plot_training_history()
    model.plot_confusion_matrix(y_test, (model.predict(X_test) > 0.5).astype(int))
    model.plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_churn_model() 