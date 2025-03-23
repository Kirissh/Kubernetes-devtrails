# Kubernetes Issue Predictor

An AI/ML model for predicting failures and issues in Kubernetes clusters by analyzing key metrics.

## Quick Start Guide for Hackathon Demo

Follow these instructions to run the project demo in 5 minutes:

### 1. Set Up the Environment

```bash
# Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### 2. Run the Demo Scripts

The project includes 3 batch files that make it easy to run different components:

```bash
# Option 1: Train the model
.\run_model.bat

# Option 2: Evaluate the model
.\run_evaluation.bat

# Option 3: Run the prediction service
.\run_deployment.bat  # Press Ctrl+C to stop
```

### 3. Explore the Results

After running the scripts, check the `results` directory to see:
- Confusion matrices
- Feature importance plots
- ROC curves
- Classification reports

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
├── run_model.bat             # Batch file to run the model training
├── run_evaluation.bat        # Batch file to run the evaluation
├── run_deployment.bat        # Batch file to run the prediction service
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Technical Details

### Data Generation

The `data_generator.py` module creates synthetic Kubernetes metrics data with the following features:
- Node CPU/memory/disk usage percentages
- Pod restart counts
- Pod CPU/memory usage
- Network I/O metrics
- HTTP error rates
- Service response times

The data includes patterns representing normal operations and four issue types:
- Pod failures
- Resource exhaustion
- Network issues
- Service disruptions

### Machine Learning Model

The core prediction model is a Random Forest Classifier with:
- 100 decision trees
- Maximum depth of 15
- Minimum samples split of 10
- Balanced class weights

Features are preprocessed using StandardScaler for normalization.

### Performance

The model achieves 100% accuracy on both training and test datasets with:
- Perfect precision and recall across all issue types
- Key predictive features identified through feature importance analysis

### Real-time Prediction

The deployment script simulates a real-time prediction service that:
1. Generates synthetic metrics at regular intervals
2. Makes predictions using the trained model
3. Displays alerts with confidence scores
4. Suggests remediation actions based on the predicted issue type

## Development Details

### Prerequisites

- Python 3.8+
- Required libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, joblib

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kubernetes-issue-predictor.git
cd kubernetes-issue-predictor

# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/MacOS

# Install dependencies
pip install -r requirements.txt
```

### Manual Execution

```bash
# Training
python -m src.main

# Evaluation
python -m src.evaluation

# Deployment service
python -m src.deployment
```

## Future Work

1. Integration with real Kubernetes API to collect actual metrics
2. Time-series analysis for more accurate predictions
3. Anomaly detection using unsupervised learning
4. Kubernetes operator for in-cluster deployment
5. Automated remediation actions based on predictions

## License

MIT License

## Contributors

Your Name
