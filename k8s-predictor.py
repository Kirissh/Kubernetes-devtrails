import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import generate_kubernetes_metrics
from model import KubernetesPredictor

def create_directories():
    """Create necessary directories for the project."""
    directories = ['data', 'models', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def main():
    print("K8s Predictor: AI/ML Model for Predicting Kubernetes Issues")
    print("=" * 60)

    # Create directories
    create_directories()

    # Generate or load data
    print("\n[1/5] Generating synthetic Kubernetes metrics data...")
    data_path = 'data/kubernetes_metrics.csv'
    if not os.path.exists(data_path):
        df = generate_kubernetes_metrics(num_samples=5000)
        df.to_csv(data_path, index=False)
        print(f"Generated data saved to {data_path}")
    else:
        df = pd.read_csv(data_path)
        print(f"Loaded existing data from {data_path}")

    # Display data summary
    print("\nData summary:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {', '.join(df.columns[:-1])}")  # Exclude the label column
    print(f"Issue distribution:\n{df['issue_type'].value_counts()}")

    # Train-test split
    print("\n[2/5] Splitting data into training and testing sets...")
    features = df.drop(['issue_type'], axis=1)
    labels = df['issue_type']
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Initialize and train the model
    print("\n[3/5] Training the prediction model...")
    predictor = KubernetesPredictor()
    predictor.train(X_train, y_train)

    # Evaluate the model
    print("\n[4/5] Evaluating model performance...")
    predictions = predictor.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"Model accuracy: {accuracy:.4f}")

    # Generate detailed classification report
    report = classification_report(y_test, predictions)
    print("\nClassification Report:")
    print(report)

    # Save classification report to file
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)

    # Save confusion matrix visualization
    print("\n[5/5] Generating visualizations...")
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=predictor.model.classes_, 
                yticklabels=predictor.model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')

    # Feature importance visualization
    feature_importances = predictor.model.feature_importances_
    feature_names = X_train.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance for Kubernetes Issue Prediction')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    
    # Save the trained model
    model_path = 'models/kubernetes_predictor.joblib'
    joblib.dump(predictor, model_path)
    print(f"\nTrained model saved to {model_path}")
    
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
