import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class KubernetesPredictor:
    """
    A machine learning model to predict Kubernetes cluster issues based on metrics.
    """
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def preprocess_data(self, X, fit_scaler=False):
        """
        Preprocess the input features.
        
        Parameters:
        - X: DataFrame with features
        - fit_scaler: Whether to fit the scaler or just transform
        
        Returns:
        - Preprocessed features
        """
        # Convert timestamp to numerical features if present
        if 'timestamp' in X.columns:
            X = X.drop('timestamp', axis=1)
            
        # Scale numerical features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled
    
    def train(self, X, y):
        """
        Train the model on the given data.
        
        Parameters:
        - X: DataFrame with features
        - y: Series with target labels
        """
        # Preprocess features
        X_processed = self.preprocess_data(X, fit_scaler=True)
        
        # Train the model
        self.model.fit(X_processed, y)
        self.is_trained = True
        
        # Print feature importance
        if hasattr(self.model, 'feature_importances_'):
            # Ensure we use the correct column names (excluding timestamp if it was dropped)
            feature_names = X.columns
            if 'timestamp' in feature_names and len(feature_names) != len(self.model.feature_importances_):
                feature_names = [col for col in feature_names if col != 'timestamp']
                
            # Create feature importance DataFrame ensuring lengths match
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            print(feature_importance.head(10))
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        - X: DataFrame with features
        
        Returns:
        - Predicted issue types
        """
        if not self.is_trained:
            raise Exception("Model has not been trained yet.")
            
        # Preprocess features
        X_processed = self.preprocess_data(X, fit_scaler=False)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        return predictions
    
    def predict_proba(self, X):
        """
        Get probability estimates for each class.
        
        Parameters:
        - X: DataFrame with features
        
        Returns:
        - Probability matrix
        """
        if not self.is_trained:
            raise Exception("Model has not been trained yet.")
            
        # Preprocess features
        X_processed = self.preprocess_data(X, fit_scaler=False)
        
        # Get probability estimates
        proba = self.model.predict_proba(X_processed)
        return proba
    
    def save(self, filepath):
        """
        Save the trained model to a file.
        
        Parameters:
        - filepath: Path to save the model file
        """
        if not self.is_trained:
            raise Exception("Cannot save untrained model.")
            
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath):
        """
        Load a trained model from a file.
        
        Parameters:
        - filepath: Path to the model file
        
        Returns:
        - Loaded model
        """
        return joblib.load(filepath)


# Example usage for anomaly detection
class KubernetesAnomalyDetector:
    """
    Extension of the predictor that focuses on anomaly detection
    by setting thresholds for metrics.
    """
    def __init__(self, threshold_config=None):
        self.threshold_config = threshold_config or {
            'node_cpu_usage_percent': 85,
            'node_memory_usage_percent': 90,
            'node_disk_usage_percent': 85,
            'pod_restart_count': 3,
            'http_error_rate': 10,
            'response_time_ms': 1000
        }
    
    def detect_anomalies(self, metrics_df):
        """
        Detect anomalies based on threshold rules.
        
        Parameters:
        - metrics_df: DataFrame with current metrics
        
        Returns:
        - DataFrame with anomaly flags
        """
        result = metrics_df.copy()
        
        # Initialize anomaly column
        result['has_anomaly'] = False
        result['anomaly_type'] = 'none'
        
        # Check CPU usage anomaly
        cpu_anomaly = metrics_df['node_cpu_usage_percent'] > self.threshold_config['node_cpu_usage_percent']
        memory_anomaly = metrics_df['node_memory_usage_percent'] > self.threshold_config['node_memory_usage_percent']
        disk_anomaly = metrics_df['node_disk_usage_percent'] > self.threshold_config['node_disk_usage_percent']
        
        # Resource exhaustion
        resource_exhaustion = cpu_anomaly | memory_anomaly | disk_anomaly
        result.loc[resource_exhaustion, 'has_anomaly'] = True
        result.loc[resource_exhaustion, 'anomaly_type'] = 'resource_exhaustion'
        
        # Pod failures
        pod_failure = metrics_df['pod_restart_count'] >= self.threshold_config['pod_restart_count']
        result.loc[pod_failure & ~resource_exhaustion, 'has_anomaly'] = True
        result.loc[pod_failure & ~resource_exhaustion, 'anomaly_type'] = 'pod_failure'
        
        # Service disruption
        service_disruption = (metrics_df['http_error_rate'] > self.threshold_config['http_error_rate']) | \
                           (metrics_df['response_time_ms'] > self.threshold_config['response_time_ms'])
        result.loc[service_disruption & ~resource_exhaustion & ~pod_failure, 'has_anomaly'] = True
        result.loc[service_disruption & ~resource_exhaustion & ~pod_failure, 'anomaly_type'] = 'service_disruption'
        
        return result 