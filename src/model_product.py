import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
from typing import Dict, List, Tuple
import os

class ProductPurchasePredictor:
    """Neural network model for product purchase prediction with collaborative filtering"""
    
    def __init__(self, input_dim: int, embedding_dim: int = 32, n_products: int = 3):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_products = n_products
        self.model = None
        self.autoencoder = None
        self.history = None
        
        # Print model configuration
        print(f"\nModel Configuration:")
        print(f"Input dimension: {input_dim}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Number of products: {n_products}")
    
    def _build_autoencoder(self) -> tf.keras.Model:
        """Build autoencoder for customer embeddings"""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(128, activation='relu')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        encoded = layers.Dense(self.embedding_dim, activation='relu')(x)
        
        # Decoder
        x = layers.Dense(128, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        # Autoencoder model
        autoencoder = models.Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def _build_model(self) -> tf.keras.Model:
        """Build the neural network model"""
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Autoencoder for customer representation
        encoded = layers.Dense(64, activation='relu')(inputs)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.3)(encoded)
        
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(encoded)
        
        # Main prediction network
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Multi-label output
        outputs = []
        for _ in range(self.n_products):
            product_output = layers.Dense(32, activation='relu')(x)
            product_output = layers.BatchNormalization()(product_output)
            product_output = layers.Dropout(0.2)(product_output)
            product_output = layers.Dense(1, activation='sigmoid')(product_output)
            outputs.append(product_output)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
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
        
        # First train autoencoder
        self.autoencoder = self._build_autoencoder()
        self.autoencoder.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs//2,
            batch_size=batch_size,
            verbose=0
        )
        
        # Get encoded features
        encoder = models.Model(
            self.autoencoder.input,
            self.autoencoder.layers[-3].output
        )
        X_train_encoded = encoder.predict(X_train)
        X_val_encoded = encoder.predict(X_val)
        
        # Build and train main model
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
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_encoded, y_train,
            validation_data=(X_val_encoded, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for multiple products"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get encoded features
        encoder = models.Model(
            self.autoencoder.input,
            self.autoencoder.layers[-3].output
        )
        X_encoded = encoder.predict(X)
        
        return self.model.predict(X_encoded)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics for each product
        metrics = {
            'accuracy': np.mean(y_pred_binary == y_test),
            'auc': tf.keras.metrics.AUC()(y_test, y_pred).numpy(),
            'confusion_matrix': tf.math.confusion_matrix(
                y_test.flatten(), y_pred_binary.flatten(), num_classes=2
            ).numpy().tolist()
        }
        
        return metrics
    
    def get_customer_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get customer embeddings from the autoencoder"""
        if self.autoencoder is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get the encoder output
        encoder = models.Model(
            self.autoencoder.input,
            self.autoencoder.layers[-3].output
        )
        
        return encoder.predict(X)
    
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