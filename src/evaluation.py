import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc
import joblib
from src.model import KubernetesPredictor
from src.data_generator import generate_kubernetes_metrics

def load_or_generate_test_data(num_samples=1000, seed=43):
    """
    Load existing test data or generate new data if not available.
    Using a different seed than the training data to simulate new data.
    """
    test_data_path = 'data/test_kubernetes_metrics.csv'
    if os.path.exists(test_data_path):
        print(f"Loading test data from {test_data_path}")
        return pd.read_csv(test_data_path)
    else:
        print(f"Generating new test data...")
        test_df = generate_kubernetes_metrics(num_samples=num_samples, seed=seed)
        test_df.to_csv(test_data_path, index=False)
        print(f"Test data saved to {test_data_path}")
        return test_df

def evaluate_model(model_path='models/kubernetes_predictor.joblib'):
    """
    Evaluate the trained model on test data.
    """
    print("\nEvaluating Kubernetes Issue Prediction Model")
    print("=" * 50)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    predictor = joblib.load(model_path)
    
    # Load or generate test data
    test_df = load_or_generate_test_data()
    
    # Split features and target
    X_test = test_df.drop(['issue_type'], axis=1)
    y_test = test_df['issue_type']
    
    # Make predictions
    print("Making predictions on test data...")
    predictions = predictor.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, predictions)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open('results/evaluation_report.txt', 'w') as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Confusion Matrix
    print("\nGenerating confusion matrix visualization...")
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Data')
    plt.tight_layout()
    plt.savefig('results/test_confusion_matrix.png')
    
    # Get probabilities for ROC curve
    try:
        proba = predictor.predict_proba(X_test)
        
        # For each class
        plt.figure(figsize=(12, 10))
        
        for i, class_name in enumerate(predictor.model.classes_):
            # OneVsRest approach - current class is positive, all others negative
            y_test_binary = (y_test == class_name).astype(int)
            class_proba = proba[:, i]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test_binary, class_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig('results/roc_curves.png')
        
    except Exception as e:
        print(f"Warning: Could not generate ROC curves: {e}")
    
    print("\nEvaluation complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    evaluate_model() 