import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import os
from feature_engineering import FeatureEngineering, FeatureConfig

class ChurnPredictor:
    """Deep learning model for customer churn prediction"""
    
    def __init__(self, input_dim: int, config: Dict[str, Any] = None):
        """
        Initialize the model
        
        Args:
            input_dim: Number of input features
            config: Model configuration parameters
        """
        self.input_dim = input_dim
        self.config = config or {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 10
        }
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self) -> models.Sequential:
        """Build the neural network architecture"""
        model = models.Sequential([
            # Input layer with L2 regularization
            layers.Dense(32, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                        input_shape=(self.input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Hidden layer with L2 regularization
            layers.Dense(16, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model with lower learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train the model"""
        # Early stopping with increased patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction with gentler factor
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=7,
            min_lr=0.000001,
            verbose=1
        )
        
        # Train with smaller batch size
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data"""
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_test = np.array(y_test).flatten()
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(y_pred == y_test),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save model weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"No model found at {filepath}")

def prepare_data_for_modeling(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for modeling"""
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: separate validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Load and prepare data
    feature_engineering = FeatureEngineering(FeatureConfig())
    X, y = feature_engineering.prepare_data()
    
    # Prepare data for modeling
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_modeling(X, y)
    
    # Initialize and train model
    model = ChurnPredictor(input_dim=X.shape[1])
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Evaluation Metrics:")
    print(metrics['classification_report'])
    
    # Save model
    model.save_model('models/churn_model.h5') 