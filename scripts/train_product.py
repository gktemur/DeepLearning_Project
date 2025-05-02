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
    """Plot training history for multi-output model"""
    plt.figure(figsize=(12, 6))

    for key in history.history:
        if 'accuracy' in key:
            linestyle = '--' if key.startswith('val_') else '-'
            plt.plot(history.history[key], linestyle=linestyle, label=key)

    plt.title('Accuracy per Output')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/product_training_history.png')
    plt.close()

def plot_confusion_matrix(cm, title='Confusion Matrix', filename='confusion_matrix.png'):
    """Plot a single confusion matrix"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'models/{filename}')
    plt.close()

def plot_roc_curve(y_test, y_pred, title='ROC Curve', filename='roc_curve.png'):
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
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f'models/{filename}')
    plt.close()

def plot_feature_distributions(X_train, y_train, feature_names):
    """Plot boxplots of feature distributions per product"""
    os.makedirs('models', exist_ok=True)

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
        explainer = ModelExplainer(model, feature_names)
        explainer.create_explainer(X_train)

        explainer.plot_global_importance(X_test, save_path='models/product_shap_global.png')

        for i in range(min(3, len(X_test))):
            try:
                explainer.plot_local_importance(X_test, index=i, save_path=f'models/product_shap_local_{i}.html')
                explainer.plot_waterfall(X_test, index=i, save_path=f'models/product_shap_waterfall_{i}.png')
                print(f"Created SHAP plots for example {i}")
            except Exception as e:
                print(f"Error creating SHAP plots for example {i}: {str(e)}")
                continue

        try:
            shap_values, explanation = explainer.explain_prediction(X_test)
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

def convert_numpy_to_python(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj

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
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize model with correct input dimension
    model = ProductPurchasePredictor(input_dim=X_train.shape[1])
    
    # Set the scaler from feature engineering
    model.scaler = feature_engineering.scaler
    
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
    
    # Convert metrics to Python native types
    metrics_python = convert_numpy_to_python(metrics)
    
    # Save metrics
    with open('models/product_model_metrics.json', 'w') as f:
        json.dump(metrics_python, f, indent=4)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: models/best_product_model.keras")
    print(f"Metrics saved to: models/product_model_metrics.json")

    # Plot confusion matrices for each product
    for i, (product_name, product_metrics) in enumerate(metrics.items(), start=1):
        cm = product_metrics['confusion_matrix']
        plot_confusion_matrix(
            cm,
            title=f'Confusion Matrix - Product {i}',
            filename=f'product_{i}_confusion_matrix.png'
        )

    # Plot ROC for the first product as example
    y_pred = model.predict(X_test.values)
    plot_roc_curve(
        y_test.values[:, 0],
        y_pred[:, 0],
        title='ROC Curve - Product 1',
        filename='product_1_roc_curve.png'
    )

    # SHAP analysis
    _ = analyze_shap(
        model.model,
        X_train.values,
        X_test.values,
        feature_engineering.category_columns
    )

    print("\nModel Evaluation:")
    for i, (product_name, product_metrics) in enumerate(metrics.items(), start=1):
        print(f"\n{product_name}:")
        print(f"  AUC: {product_metrics['auc']:.4f}")
        print(f"  Accuracy: {product_metrics['accuracy']:.4f}")
        print(f"  Confusion Matrix:\n{np.array(product_metrics['confusion_matrix'])}")

    print("\nPlots and analysis saved in the 'models' directory:")
    print("- product_training_history.png")
    print("- product_1_roc_curve.png")
    print("- product_[1-3]_confusion_matrix.png")
    print("- product_feature_distributions.png")
    print("- product_shap_global.png")
    print("- product_shap_local_*.html")
    print("- product_shap_waterfall_*.png")
    print("- product_shap_explanation.json")

if __name__ == "__main__":
    main()
