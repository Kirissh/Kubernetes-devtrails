import pandas as pd
import numpy as np
import joblib
import time
import argparse
import json
from datetime import datetime

def generate_live_metrics():
    """
    Generate a single sample of simulated live Kubernetes metrics.
    In a real deployment, this would be replaced by actual metrics collection
    from a Kubernetes cluster using tools like Prometheus.
    """
    # Generate a random normal sample with occasional anomalies
    is_anomaly = np.random.random() < 0.15  # 15% chance of anomaly
    
    # Base metrics - normal operation
    metrics = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'node_cpu_usage_percent': np.random.normal(40, 15),
        'node_memory_usage_percent': np.random.normal(50, 15),
        'node_disk_usage_percent': np.random.normal(60, 10),
        'pod_restart_count': np.random.poisson(0.1),
        'pod_cpu_usage_percent': np.random.normal(30, 10),
        'pod_memory_usage_percent': np.random.normal(40, 15),
        'network_receive_bytes': np.random.normal(5000, 2000),
        'network_transmit_bytes': np.random.normal(3000, 1000),
        'http_error_rate': np.random.beta(1, 50) * 100,
        'response_time_ms': np.random.gamma(2, 50)
    }
    
    # Introduce anomalies based on random issue type
    if is_anomaly:
        issue_type = np.random.choice(['pod_failure', 'resource_exhaustion', 'network_issue', 'service_disruption'])
        
        if issue_type == 'pod_failure':
            metrics['pod_restart_count'] = np.random.randint(3, 10)
            metrics['pod_cpu_usage_percent'] = np.random.normal(85, 10)
            metrics['pod_memory_usage_percent'] = np.random.normal(90, 5)
            
        elif issue_type == 'resource_exhaustion':
            metrics['node_cpu_usage_percent'] = np.random.normal(92, 5)
            metrics['node_memory_usage_percent'] = np.random.normal(95, 3)
            metrics['node_disk_usage_percent'] = np.random.normal(90, 5)
            
        elif issue_type == 'network_issue':
            metrics['network_receive_bytes'] = np.random.normal(500, 300)
            metrics['network_transmit_bytes'] = np.random.normal(300, 200)
            metrics['http_error_rate'] = np.random.beta(5, 5) * 100
            
        elif issue_type == 'service_disruption':
            metrics['response_time_ms'] = np.random.gamma(20, 30)
            metrics['http_error_rate'] = np.random.beta(8, 2) * 100
    
    # Ensure all percentages are between 0 and 100
    for key in metrics:
        if 'percent' in key:
            metrics[key] = max(0, min(100, metrics[key]))
    
    # Ensure counts are non-negative integers
    if 'pod_restart_count' in metrics:
        metrics['pod_restart_count'] = max(0, int(metrics['pod_restart_count']))
    
    # Ensure byte counts and response times