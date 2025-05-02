import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
from typing import Dict, List, Tuple
import os
import joblib

class ProductPurchasePredictor:
    """Neural network model for predicting new product purchase probabilities"""
    
    def __init__(self, input_dim: int = 8, embedding_dim: int = 32, n_products: int = 3):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_products = n_products
        self.model = None
        self.history = None
        self.scaler = None
        
        # Print model configuration
        print(f"\nModel Configuration:")
        print(f"Input dimension: {input_dim}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Number of products: {n_products}")
    
    def _build_model(self) -> tf.keras.Model:
        """Build the neural network model"""
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Feature extraction
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Product-specific branches
        outputs = []
        for _ in range(self.n_products):
            # Product-specific features
            product_branch = layers.Dense(16, activation='relu')(x)
            product_branch = layers.BatchNormalization()(product_branch)
            product_branch = layers.Dropout(0.2)(product_branch)
            
            # Product purchase probability
            product_output = layers.Dense(1, activation='sigmoid')(product_branch)
            outputs.append(product_output)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=['binary_crossentropy'] * self.n_products,
            metrics=['accuracy'] * self.n_products
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              cost_ratio: float = 2.0):
        """Train the model with cost-sensitive learning"""
        # Verify input shapes
        print("\nTraining data shapes:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        
        # Build model
        self.model = self._build_model()
        
        # Calculate class weights
        class_weights = {
            0: 1.0,  # Non-purchasing class
            1: cost_ratio  # Purchasing class
        }
        
        # Create callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
            callbacks.ModelCheckpoint(
                filepath='models/best_product_model.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]
        
        # Split y_train and y_val into separate arrays for each product
        y_train_split = [y_train[:, i] for i in range(self.n_products)]
        y_val_split = [y_val[:, i] for i in range(self.n_products)]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train_split,
            validation_data=(X_val, y_val_split),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save the final model
        self.save_model()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for multiple products"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get predictions for each product
        predictions = self.model.predict(X)
        
        # Combine predictions into a single array
        return np.column_stack(predictions)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics for each product
        metrics = {}
        for i in range(self.n_products):
            product_metrics = {
                'accuracy': tf.keras.metrics.binary_accuracy(y_test[:, i], y_pred[:, i]).numpy().mean(),
                'auc': tf.keras.metrics.AUC()(y_test[:, i], y_pred[:, i]).numpy(),
                'confusion_matrix': tf.math.confusion_matrix(
                    y_test[:, i], 
                    (y_pred[:, i] > 0.5).astype(int),
                    num_classes=2
                ).numpy().tolist()
            }
            metrics[f'product_{i+1}'] = product_metrics
        
        return metrics
    
    def save_model(self):
        """Save the trained model"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        model_path = 'models/best_product_model.keras'
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save the scaler if it exists
        if self.scaler is not None:
            scaler_path = 'models/product_scaler.joblib'
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
        else:
            print("Warning: No scaler to save")
    
    def find_similar_customers(self, customer_embeddings: np.ndarray,
                             target_customer_idx: int,
                             n_similar: int = 5) -> List[int]:
        """Find similar customers using cosine similarity"""
        # Get target customer embedding
        target_embedding = customer_embeddings[target_customer_idx]
        
        # Calculate cosine similarities
        similarities = np.dot(customer_embeddings, target_embedding) / (
            np.linalg.norm(customer_embeddings, axis=1) * np.linalg.norm(target_embedding)
        )
        
        # Get top n similar customers (excluding the target customer)
        similar_indices = np.argsort(similarities)[-n_similar-1:-1][::-1]
        
        return similar_indices.tolist()
    
    def get_product_recommendations(self, X: np.ndarray, 
                                  threshold: float = 0.5) -> List[List[int]]:
        """Get product recommendations for each customer"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get predictions
        predictions = self.predict(X)
        
        # Get recommended products for each customer
        recommendations = []
        for pred in predictions:
            recommended_products = np.where(pred > threshold)[0].tolist()
            recommendations.append(recommended_products)
            
        return recommendations 