import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
from src.model import KubernetesPredictor

def simulate_live_metrics():
    """
    Simulate gathering of live Kubernetes metrics.
    In a real scenario, this would connect to the K8s API server.
    """
    # Simulate a single node with multiple pods
    num_pods = np.random.randint(5, 20)
    
    # Node metrics
    node_cpu = np.random.normal(60, 20)
    node_memory = np.random.normal(55, 15)
    node_disk = np.random.normal(70, 10)
    
    # Network metrics
    network_rx = np.random.normal(4000, 1000)
    network_tx = np.random.normal(2500, 800)
    http_error_rate = np.random.beta(1, 20) * 100
    response_time = np.random.gamma(2, 100)
    
    # Introduce anomalies occasionally (10% chance)
    if np.random.random() < 0.1:
        anomaly_type = np.random.choice(['cpu', 'memory', 'disk', 'network', 'pod'])
        
        if anomaly_type == 'cpu':
            node_cpu = np.random.normal(95, 3)
        elif anomaly_type == 'memory':
            node_memory = np.random.normal(92, 5)
        elif anomaly_type == 'disk':
            node_disk = np.random.normal(95, 3)
        elif anomaly_type == 'network':
            http_error_rate = np.random.beta(5, 2) * 100
            response_time = np.random.gamma(10, 100)
    
    # Create metrics dictionary
    metrics = {
        'timestamp': str(datetime.now()),
        'node_cpu_usage_percent': max(0, min(100, node_cpu)),
        'node_memory_usage_percent': max(0, min(100, node_memory)),
        'node_disk_usage_percent': max(0, min(100, node_disk)),
        'pod_restart_count': np.random.poisson(0.3),
        'pod_cpu_usage_percent': max(0, min(100, np.random.normal(40, 15))),
        'pod_memory_usage_percent': max(0, min(100, np.random.normal(50, 15))),
        'network_receive_bytes': max(0, network_rx),
        'network_transmit_bytes': max(0, network_tx),
        'http_error_rate': max(0, http_error_rate),
        'response_time_ms': max(0, response_time)
    }
    
    return metrics

def run_prediction_service(model_path='models/kubernetes_predictor.joblib', interval=5):
    """
    Continuous prediction service that monitors K8s cluster metrics
    and predicts potential issues.
    
    Parameters:
    - model_path: Path to the trained model
    - interval: Time between predictions in seconds
    """
    print(f"Loading prediction model from {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    predictor = joblib.load(model_path)
    print("Prediction service started. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Get current metrics
            metrics = simulate_live_metrics()
            
            # Convert to DataFrame for model input
            metrics_df = pd.DataFrame([metrics])
            
            # Make prediction
            prediction = predictor.predict(metrics_df)[0]
            probabilities = predictor.predict_proba(metrics_df)[0]
            
            # Find the class with highest probability
            class_names = predictor.model.classes_
            highest_prob = np.max(probabilities)
            predicted_class = class_names[np.argmax(probabilities)]
            
            # Print the prediction with a timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Current Prediction:")
            print(f"Node CPU: {metrics['node_cpu_usage_percent']:.1f}%, Memory: {metrics['node_memory_usage_percent']:.1f}%, Disk: {metrics['node_disk_usage_percent']:.1f}%")
            
            if predicted_class == 'none':
                print("Status: Normal operation (no issues detected)")
                print(f"Confidence: {highest_prob:.2%}")
            else:
                # Alert if there's an issue detected
                print(f"⚠️ ALERT: Potential {predicted_class.replace('_', ' ')} detected!")
                print(f"Confidence: {highest_prob:.2%}")
                print("Recommended actions:")
                
                if predicted_class == 'resource_exhaustion':
                    print("- Scale up the affected resources")
                    print("- Check for resource-intensive workloads")
                elif predicted_class == 'pod_failure':
                    print("- Check pod logs for errors")
                    print("- Verify pod configuration")
                elif predicted_class == 'network_issue':
                    print("- Check network policies")
                    print("- Verify service configurations")
                elif predicted_class == 'service_disruption':
                    print("- Verify service dependencies")
                    print("- Check for external service outages")
            
            # Wait for the next interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nPrediction service stopped.")

if __name__ == "__main__":
    run_prediction_service() 