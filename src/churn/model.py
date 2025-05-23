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
from sklearn.utils import class_weight
from src.churn.feature_engineering import FeatureEngineering, FeatureConfig

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
        """Build the improved neural network architecture"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile model with adjusted learning rate and class weights
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
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
        """Train the model with improved handling of class imbalance"""
        # Calculate class weights with higher emphasis on minority class
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        # Increase weight for minority class
        weights[1] *= 3  # Triple the weight for churn class
        class_weights = dict(zip(np.unique(y_train), weights))
        print("Computed class weights:", class_weights)

        # Early stopping with improved patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=30,
            restore_best_weights=True,
            mode='max'
        )

        # Learning rate scheduler
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.2,
            patience=15,
            min_lr=1e-7,
            mode='max'
        )

        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )

        # Train model with callbacks
        self.history = self.model.fit(
            X_train.to_numpy(),
            y_train.to_numpy(),
            validation_data=(X_val.to_numpy(), y_val.to_numpy()),
            epochs=300,  # Increased epochs
            batch_size=32,  # Reduced batch size
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1,
            class_weight=class_weights
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
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"No model found at {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("No training history available")
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

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
    
    # Plot results
    model.plot_training_history()
    model.plot_confusion_matrix(y_test, (model.predict(X_test) > 0.5).astype(int))
    model.plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
    
    # Save model
    #model.save_model('models/churn_model.h5') 
    model.save_model('models/churn_model.weights.h5')
    y_pred = model.predict(X_test)
    print("Tahminler:", y_pred[:10])
    print(pd.Series(y_train).value_counts())



