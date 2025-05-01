import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import Tuple, Dict
import os

class ModelExplainer:
    """Class for model explainability using SHAP"""
    
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def create_explainer(self, X_train: np.ndarray):
        """Create SHAP explainer"""
        # Convert to numpy array if it's a DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
            
        # Create explainer with flattened model output
        self.explainer = shap.KernelExplainer(
            lambda x: self.model.predict(x).flatten(),  # Flatten TensorFlow output for SHAP
            X_train[:100]  # Use first 100 samples as background
        )
        
    def explain_prediction(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Explain a single prediction"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
            
        # Convert to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle single output case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Get feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create explanation dictionary
        explanation = {
            'shap_values': shap_values.tolist(),
            'feature_importance': dict(zip(self.feature_names, feature_importance.tolist())),
            'base_value': float(self.explainer.expected_value)
        }
        
        return shap_values, explanation
    
    def plot_global_importance(self, X: np.ndarray, save_path: str = None):
        """Plot global feature importance"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
            
        # Convert to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle single output case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Plot summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_local_importance(self, X: np.ndarray, index: int, save_path: str = None):
        """Plot local feature importance for a single prediction"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
            
        # Convert to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Get the specific instance
        instance = X[index]
        
        # Calculate SHAP values for this instance
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        
        # Handle single output case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Ensure shap_values is 1D
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
            
        # Create HTML force plot
        force_plot = shap.force_plot(
            self.explainer.expected_value,
            shap_values,
            instance,
            feature_names=self.feature_names,
            matplotlib=False
        )
        
        if save_path:
            # Save as HTML
            shap.save_html(save_path.replace('.png', '.html'), force_plot)
        else:
            return force_plot
    
    def plot_waterfall(self, X: np.ndarray, index: int, save_path: str = None):
        """Plot waterfall plot for a single prediction"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
            
        # Convert to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Get the specific instance
        instance = X[index]
        
        # Calculate SHAP values for this instance
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        
        # Handle single output case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Ensure shap_values is 1D
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
            
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
                data=instance,
                feature_names=self.feature_names
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 