from src.feature_engineering_iade import IadeFeatureEngineering
from src.model_iade import IadeRiskPredictor
from src.explainability import ModelExplainer
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
import json

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
    plt.savefig('models/iade_training_history.png')
    plt.close()

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('models/iade_confusion_matrix.png')
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
    plt.savefig('models/iade_roc_curve.png')
    plt.close()

def plot_feature_distributions(X_train, y_train):
    """Plot feature distributions for risky and non-risky orders"""
    plt.figure(figsize=(15, 5))
    
    for i, feature in enumerate(X_train.columns):
        plt.subplot(1, 3, i+1)
        sns.boxplot(x=y_train, y=X_train[feature])
        plt.title(f'Distribution of {feature}')
        plt.xlabel('Is Risky')
        plt.ylabel(feature)
    
    plt.tight_layout()
    plt.savefig('models/iade_feature_distributions.png')
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
            save_path='models/iade_shap_global.png'
        )
        
        # Plot local importance for a few examples
        for i in range(min(3, len(X_test))):
            try:
                # Save force plot as HTML
                explainer.plot_local_importance(
                    X_test,
                    index=i,
                    save_path=f'models/iade_shap_local_{i}.html'
                )
                
                # Save waterfall plot as PNG
                explainer.plot_waterfall(
                    X_test,
                    index=i,
                    save_path=f'models/iade_shap_waterfall_{i}.png'
                )
                print(f"Created SHAP plots for example {i}")
            except Exception as e:
                print(f"Error creating SHAP plots for example {i}: {str(e)}")
                continue
        
        # Get SHAP values for test set
        try:
            shap_values, explanation = explainer.explain_prediction(X_test)
            
            # Save explanation to file
            with open('models/iade_shap_explanation.json', 'w') as f:
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
    feature_engineering = IadeFeatureEngineering()
    
    # Prepare data
    X_train, X_test, y_train, y_test = feature_engineering.prepare_data()
    
    # Plot feature distributions
    plot_feature_distributions(X_train, y_train)
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize and train model with cost-sensitive learning
    model = IadeRiskPredictor(input_dim=X_train.shape[1])
    model.train(
        X_train=X_train.values,
        y_train=y_train.values,
        X_val=X_val.values,
        y_val=y_val.values,
        epochs=100,
        batch_size=32,
        cost_ratio=5.0  # Higher weight for risky orders
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
        X_train.columns.tolist()
    )
    
    print("\nModel Evaluation:")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    
    print("\nPlots and analysis saved in the 'models' directory:")
    print("- iade_training_history.png")
    print("- iade_confusion_matrix.png")
    print("- iade_roc_curve.png")
    print("- iade_feature_distributions.png")
    print("- iade_shap_global.png")
    print("- iade_shap_local_*.html (HTML files for force plots)")
    print("- iade_shap_waterfall_*.png (Waterfall plots)")
    print("- iade_shap_explanation.json")

if __name__ == "__main__":
    main() 