import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import joblib
import os

class IadeRiskPredictor:
    """Deep learning model for return risk prediction"""
    
    def __init__(self, input_dim: int):
        self.model = self._build_model(input_dim)
        self.history = None

    def _build_model(self, input_dim: int) -> Sequential:
        """Build the neural network model"""
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              cost_ratio: float = 5.0):
        """Train the model with callbacks and cost-sensitive learning"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath='models/best_iade_model.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Calculate class weights with cost-sensitive learning
        # cost_ratio: ratio of cost of false negative to false positive
        # Higher cost_ratio means we care more about missing risky orders
        n_samples = len(y_train)
        n_risky = np.sum(y_train == 1)
        n_safe = n_samples - n_risky
        
        # Calculate weights based on cost ratio
        weight_risky = n_samples / (2 * n_risky) * cost_ratio
        weight_safe = n_samples / (2 * n_safe)
        
        class_weights = {
            0: weight_safe,
            1: weight_risky
        }
        
        print(f"\nClass weights with cost ratio {cost_ratio}:")
        print(f"Safe orders weight: {weight_safe:.2f}")
        print(f"Risky orders weight: {weight_risky:.2f}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred_binary)
        auc = roc_auc_score(y_test, y_pred)
        
        metrics = {
            'confusion_matrix': cm.tolist(),
            'auc': float(auc),
            'accuracy': float(np.mean(y_pred_binary == y_test))
        }
        
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def save_model(self, filepath: str):
        """Save the model"""
        self.model.save(filepath)

    def load_model(self, filepath: str):
        """Load the model"""
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}") 