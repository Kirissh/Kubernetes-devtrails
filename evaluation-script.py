import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from model import KubernetesPredictor

def load_test_data(filepath='data/kubernetes_metrics.csv', test_size=0.2):
    """
    Load and prepare test data
    """
    # Load the complete dataset
    df = pd.read_csv(filepath)
    
    # Take the last test_size portion as test data to simulate future data
    test_idx = int(len(df) * (1 - test_size))
    test_data = df.iloc[test_idx:]
    
    X_test = test_data.drop('issue_type', axis=1)
    y_test = test_data['issue_type']
    
    return X_test, y_test

def evaluate_model(model_path='models/kubernetes_predictor.joblib'):
    """
    Evaluate a trained model on test data
    """
    # Load the model
    predictor = joblib.load(model_path)
    
    # Load test data
    X_test, y_test = load_test_data()
    
    print(f"Evaluating model on {len(X_test)} test samples...")
    
    # Make predictions
    y_pred = predictor.predict(X_test)
    y_pred_proba = predictor.predict_proba(X_test)
    
    # Basic accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open('results/evaluation_report.txt', 'w') as f:
        f.write(f"Model accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=predictor.model.classes_,
                yticklabels=predictor.model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Data')
    plt.tight_layout()
    plt.savefig('results/test_confusion_matrix.png')
    
    # ROC curve for multiclass
    n_classes = len(predictor.model.classes_)
    
    # Binarize the labels for ROC curve
    y_test_bin = label_binarize(y_test, classes=predictor.model.classes_)
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(predictor.model.classes_):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve of {class_name} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Issue Prediction')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curves.png')
    
    return accuracy, report

if __name__ == "__main__":
    evaluate_model()
