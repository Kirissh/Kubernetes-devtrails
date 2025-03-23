# Kubernetes Issue Predictor

An AI/ML model for predicting failures and issues in Kubernetes clusters by analyzing metrics.

## Project Overview

This project aims to predict potential issues in Kubernetes clusters before they occur, including:
- Node or pod failures
- Resource exhaustion (CPU, memory, disk)
- Network or connectivity issues
- Service disruptions

The solution uses machine learning to analyze cluster metrics and identify patterns that precede failures.

## Project Structure

```
kubernetes-issue-predictor/
├── data/                     # Directory for datasets
├── models/                   # Directory for saved ML models
├── results/                  # Directory for output results and visualizations
├── src/                      # Source code
│   ├── __init__.py           # Package initialization
│   ├── data_generator.py     # Module to generate synthetic K8s metrics
│   ├── model.py              # ML model implementation
│   ├── main.py               # Main training script
│   ├── evaluation.py         # Model evaluation script
│   └── deployment.py         # Deployment/prediction service script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kubernetes-issue-predictor.git
cd kubernetes-issue-predictor
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the prediction model using synthetic data:

```bash
python -m src.main
```

This will:
1. Generate synthetic Kubernetes metrics data
2. Train a Random Forest classifier
3. Evaluate the model performance
4. Save the trained model and visualizations

### Evaluating the Model

To evaluate the trained model on a separate test dataset:

```bash
python -m src.evaluation
```

### Running the Prediction Service

To run the real-time prediction service (which simulates metrics collection):

```bash
python -m src.deployment
```

Press Ctrl+C to stop the service.

### Generating Synthetic Data

To generate a new dataset of synthetic Kubernetes metrics:

```bash
python -m src.data_generator
```

## Model Details

The prediction model uses a Random Forest classifier with the following metrics as input features:
- Node CPU usage (%)
- Node memory usage (%)
- Node disk usage (%)
- Pod restart count
- Pod CPU and memory usage (%)
- Network I/O (bytes)
- HTTP error rate
- Service response time (ms)

The model predicts one of these issue types:
- `none` (normal operation)
- `resource_exhaustion`
- `pod_failure`
- `network_issue`
- `service_disruption`

## Performance

The model typically achieves:
- Accuracy: ~95%
- Precision: >90% for most issue types
- Recall: >85% for most issue types

See the `results/` directory for detailed metrics and visualizations.

## Future Improvements

1. Integration with real Kubernetes API to collect actual metrics
2. Time-series analysis for more accurate predictions
3. Additional features like log sentiment analysis
4. Anomaly detection using unsupervised learning
5. Kubernetes operator deployment

## Kubernetes Deployment (Optional)

For deploying this solution on a Kubernetes cluster:

1. Build the Docker image:
```bash
docker build -t k8s-predictor:latest .
```

2. Deploy using the provided manifests:
```bash
kubectl apply -f kubernetes/
```

## License

MIT License

## Author

Your Name 