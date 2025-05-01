from src.feature_engineering_product import ProductFeatureEngineering
from src.model_product import ProductPurchasePredictor
from src.explainability import ModelExplainer
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
import json
import pandas as pd

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/product_training_history.png')
    plt.close()

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('models/product_confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_test, y_pred):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('models/product_roc_curve.png')
    plt.close()

def plot_feature_distributions(X_train, y_train, feature_names):
    """Plot boxplots of feature distributions per product"""
    os.makedirs('models', exist_ok=True)

    # Convert back to DataFrame if needed
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    if not isinstance(y_train, pd.DataFrame):
        y_train = pd.DataFrame(y_train)

    for product in y_train.columns:
        plt.figure(figsize=(15, 5))
        for i, feature in enumerate(feature_names):
            plt.subplot(1, len(feature_names), i + 1)
            sns.boxplot(x=y_train[product], y=X_train[feature])
            plt.title(f"{product} - {feature}")
            plt.xlabel("Target")
            plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(f"models/feature_distribution_{product}.png")
        plt.close()

def analyze_shap(model, X_train, X_test, feature_names):
    """Perform SHAP analysis"""
    try:
        # Create explainer
        explainer = ModelExplainer(model, feature_names)
        explainer.create_explainer(X_train)
        
        # Plot global feature importance
        explainer.plot_global_importance(
            X_test,
            save_path='models/product_shap_global.png'
        )
        
        # Plot local importance for a few examples
        for i in range(min(3, len(X_test))):
            try:
                # Save force plot as HTML
                explainer.plot_local_importance(
                    X_test,
                    index=i,
                    save_path=f'models/product_shap_local_{i}.html'
                )
                
                # Save waterfall plot as PNG
                explainer.plot_waterfall(
                    X_test,
                    index=i,
                    save_path=f'models/product_shap_waterfall_{i}.png'
                )
                print(f"Created SHAP plots for example {i}")
            except Exception as e:
                print(f"Error creating SHAP plots for example {i}: {str(e)}")
                continue
        
        # Get SHAP values for test set
        try:
            shap_values, explanation = explainer.explain_prediction(X_test)
            
            # Save explanation to file
            with open('models/product_shap_explanation.json', 'w') as f:
                json.dump(explanation, f, indent=4)
            print("Created SHAP explanation file")
        except Exception as e:
            print(f"Error creating SHAP explanation: {str(e)}")
            explanation = None
        
        return explainer
    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        return None

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize feature engineering
    feature_engineering = ProductFeatureEngineering()
    
    # Prepare data
    X_train, X_test, y_train, y_test = feature_engineering.prepare_data()
    
    # Print data shapes for verification
    print("\nData shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Plot feature distributions
    plot_feature_distributions(X_train.values, y_train.values, feature_engineering.category_columns)
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize model with correct input dimension
    model = ProductPurchasePredictor(input_dim=X_train.shape[1])
    
    # Train model with cost-sensitive learning
    model.train(
        X_train=X_train.values,
        y_train=y_train.values,
        X_val=X_val.values,
        y_val=y_val.values,
        epochs=100,
        batch_size=32,
        cost_ratio=2.0  # Higher weight for purchasing class
    )
    
    # Plot training history
    plot_training_history(model.history)
    
    # Evaluate model
    metrics = model.evaluate(X_test.values, y_test.values)
    
    # Plot confusion matrix
    plot_confusion_matrix(np.array(metrics['confusion_matrix']))
    
    # Plot ROC curve
    y_pred = model.predict(X_test.values)
    plot_roc_curve(y_test.values, y_pred)
    
    # Perform SHAP analysis
    explainer = analyze_shap(
        model.model,
        X_train.values,
        X_test.values,
        feature_engineering.category_columns
    )
    
    # Get customer embeddings and find similar customers
    customer_embeddings = model.get_customer_embeddings(X_test.values)
    similar_customers = model.find_similar_customers(
        customer_embeddings,
        target_customer_idx=0,  # Example: first customer in test set
        n_similar=5
    )
    
    print("\nModel Evaluation:")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    
    print("\nSimilar Customers (indices):")
    print(similar_customers)
    
    print("\nPlots and analysis saved in the 'models' directory:")
    print("- product_training_history.png")
    print("- product_confusion_matrix.png")
    print("- product_roc_curve.png")
    print("- product_feature_distributions.png")
    print("- product_shap_global.png")
    print("- product_shap_local_*.html (HTML files for force plots)")
    print("- product_shap_waterfall_*.png (Waterfall plots)")
    print("- product_shap_explanation.json")

if __name__ == "__main__":
    main() 