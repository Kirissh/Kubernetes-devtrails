import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_kubernetes_metrics(num_samples=5000, seed=42):
    """
    Generate synthetic Kubernetes metrics data for training the prediction model.
    
    Parameters:
    - num_samples: Number of data points to generate
    - seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with synthetic metrics and issue labels
    """
    np.random.seed(seed)
    
    # Initialize DataFrame
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=i*5) for i in range(num_samples)]
    
    data = {
        'timestamp': timestamps,
        'node_cpu_usage_percent': np.zeros(num_samples),
        'node_memory_usage_percent': np.zeros(num_samples),
        'node_disk_usage_percent': np.zeros(num_samples),
        'pod_restart_count': np.zeros(num_samples),
        'pod_cpu_usage_percent': np.zeros(num_samples),
        'pod_memory_usage_percent': np.zeros(num_samples),
        'network_receive_bytes': np.zeros(num_samples),
        'network_transmit_bytes': np.zeros(num_samples),
        'http_error_rate': np.zeros(num_samples),
        'response_time_ms': np.zeros(num_samples),
        'issue_type': ['none'] * num_samples
    }
    
    df = pd.DataFrame(data)
    
    # Generate normal operating conditions
    df['node_cpu_usage_percent'] = np.random.normal(40, 15, num_samples)
    df['node_memory_usage_percent'] = np.random.normal(50, 15, num_samples)
    df['node_disk_usage_percent'] = np.random.normal(60, 10, num_samples)
    df['pod_restart_count'] = np.random.poisson(0.1, num_samples)
    df['pod_cpu_usage_percent'] = np.random.normal(30, 10, num_samples)
    df['pod_memory_usage_percent'] = np.random.normal(40, 15, num_samples)
    df['network_receive_bytes'] = np.random.normal(5000, 2000, num_samples)
    df['network_transmit_bytes'] = np.random.normal(3000, 1000, num_samples)
    df['http_error_rate'] = np.random.beta(1, 50, num_samples) * 100  # Mostly low values
    df['response_time_ms'] = np.random.gamma(2, 50, num_samples)
    
    # Ensure all percentages are between 0 and 100
    for col in [col for col in df.columns if 'percent' in col]:
        df[col] = df[col].clip(0, 100)
    
    # Make sure count data is non-negative and integer
    df['pod_restart_count'] = df['pod_restart_count'].clip(0).astype(int)
    
    # Ensure byte counts and response times are positive
    df['network_receive_bytes'] = df['network_receive_bytes'].clip(0)
    df['network_transmit_bytes'] = df['network_transmit_bytes'].clip(0)
    df['response_time_ms'] = df['response_time_ms'].clip(0)
    
    # Introduce pod failure patterns (about 5% of data)
    pod_failure_idx = np.random.choice(
        num_samples, size=int(num_samples * 0.05), replace=False
    )
    df.loc[pod_failure_idx, 'pod_restart_count'] += np.random.randint(3, 10, size=len(pod_failure_idx))
    df.loc[pod_failure_idx, 'pod_cpu_usage_percent'] = np.random.normal(85, 10, size=len(pod_failure_idx))
    df.loc[pod_failure_idx, 'pod_memory_usage_percent'] = np.random.normal(90, 5, size=len(pod_failure_idx))
    df.loc[pod_failure_idx, 'issue_type'] = 'pod_failure'
    
    # Introduce resource exhaustion patterns (about 7% of data)
    resource_exhaust_idx = np.random.choice(
        [i for i in range(num_samples) if i not in pod_failure_idx], 
        size=int(num_samples * 0.07), 
        replace=False
    )
    df.loc[resource_exhaust_idx, 'node_cpu_usage_percent'] = np.random.normal(92, 5, size=len(resource_exhaust_idx))
    df.loc[resource_exhaust_idx, 'node_memory_usage_percent'] = np.random.normal(95, 3, size=len(resource_exhaust_idx))
    df.loc[resource_exhaust_idx, 'node_disk_usage_percent'] = np.random.normal(90, 5, size=len(resource_exhaust_idx))
    df.loc[resource_exhaust_idx, 'issue_type'] = 'resource_exhaustion'
    
    # Introduce network issues (about 4% of data)
    network_issue_idx = np.random.choice(
        [i for i in range(num_samples) if i not in pod_failure_idx and i not in resource_exhaust_idx], 
        size=int(num_samples * 0.04), 
        replace=False
    )
    df.loc[network_issue_idx, 'network_receive_bytes'] = np.random.normal(500, 300, size=len(network_issue_idx))
    df.loc[network_issue_idx, 'network_transmit_bytes'] = np.random.normal(300, 200, size=len(network_issue_idx))
    df.loc[network_issue_idx, 'http_error_rate'] = np.random.beta(5, 5, size=len(network_issue_idx)) * 100
    df.loc[network_issue_idx, 'issue_type'] = 'network_issue'
    
    # Introduce service disruption (about 4% of data)
    service_disruption_idx = np.random.choice(
        [i for i in range(num_samples) if i not in pod_failure_idx and 
                                          i not in resource_exhaust_idx and 
                                          i not in network_issue_idx], 
        size=int(num_samples * 0.04), 
        replace=False
    )
    df.loc[service_disruption_idx, 'response_time_ms'] = np.random.gamma(20, 30, size=len(service_disruption_idx))
    df.loc[service_disruption_idx, 'http_error_rate'] = np.random.beta(8, 2, size=len(service_disruption_idx)) * 100
    df.loc[service_disruption_idx, 'issue_type'] = 'service_disruption'
    
    # Convert timestamp to string for easier CSV handling
    df['timestamp'] = df['timestamp'].astype(str)
    
    return df


if __name__ == "__main__":
    # Generate a sample dataset and save it to CSV
    df = generate_kubernetes_metrics(num_samples=5000)
    df.to_csv("data/kubernetes_metrics.csv", index=False)
    print("Generated synthetic Kubernetes metrics data with shape:", df.shape)
    print("Issue distribution:")
    print(df['issue_type'].value_counts()) 