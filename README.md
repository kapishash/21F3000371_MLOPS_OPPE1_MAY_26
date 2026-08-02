# Stock Movement Predictor - MLOps OPPE-1

## Problem Statement

Build an end-to-end MLOps pipeline for predicting stock movement in the next 5 minutes using minute-level stock data.

The project implements:

- Data Versioning using DVC
- Feature Engineering and Feature Store using Feast
- Model Training using LightGBM
- Experiment Tracking and Model Registry using MLflow
- Continuous Integration using GitHub Actions and CML
- Deployment-ready ML pipeline on Google Cloud Platform

---

# Project Objective

The goal is to predict whether a stock price will increase or decrease after 5 minutes.

Target:

```
1 → Stock close price after 5 minutes > current close price

0 → Otherwise
```

Features used:

| Feature | Description |
|---|---|
| rolling_avg_10 | Moving average of close price over previous 10 minutes |
| volume_sum_10 | Total trading volume over previous 10 minutes |
| stock_name | Stock identifier |

---

# Project Structure

```
.
├── data/
│   ├── v0/
│   └── v1/
│
├── feast_repo/
│   ├── feature_store.yaml
│   ├── feature_views.py
│   └── data/
│
├── models/
│   ├── model.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── get_training_data.py
│
├── .github/
│   └── workflows/
│       └── cml.yaml
│
├── requirements.txt
├── dvc.yaml
└── README.md
```

---

# Environment Setup

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Data Versioning using DVC

Initialize DVC:

```bash
dvc init
```

Configure Google Cloud Storage remote:

```bash
dvc remote add -d gcs_remote gs://<bucket-name>
```

Track datasets:

```bash
dvc add data/v0
dvc add data/v1
```

Commit DVC files:

```bash
git add .
git commit -m "Track datasets using DVC"
```

Push data to GCS:

```bash
dvc push
```

Restore data:

```bash
dvc pull
```

---

# Data Preparation

The raw stock files are processed to create:

- rolling_avg_10
- volume_sum_10
- target column

The dataset is sorted chronologically before splitting.

Run:

## Iteration 1 - v0 Data

```bash
python src/prepare_data.py --version v0
```

Output:

```
feast_repo/data/train.parquet
feast_repo/data/test.parquet
```

---

## Iteration 2 - Merged v0 + v1 Data

```bash
python src/prepare_data.py --version merged
```

---

# Feast Feature Store

Feast is used for storing and retrieving point-in-time correct features.

Initialize Feast:

```bash
cd feast_repo
feast apply
```

Materialize features:

```bash
feast materialize-incremental $(date +%Y-%m-%d)
```

Retrieve training features:

```bash
python ../src/get_training_data.py
```

Features served:

```
stock_features:
    rolling_avg_10
    volume_sum_10
```

Entity:

```
stock_name
```

---

# Model Training

Algorithm:

```
LightGBM Binary Classifier
```

Training includes hyperparameter tuning.

Configurations tested:

```
n_estimators:
100, 200

learning_rate:
0.05, 0.1

max_depth:
5,6,8
```

Run training:

## v0 Training

```bash
python src/train.py --iteration v0
```

## Merged Training

```bash
python src/train.py --iteration merged
```

The best model is saved:

```
models/model.pkl
models/label_encoder.pkl
```

---

# MLflow Experiment Tracking

MLflow tracks:

- Parameters
- Accuracy
- Precision
- Recall
- F1 Score
- Model artifacts

Start MLflow UI:

```bash
mlflow ui
```

Open:

```
http://localhost:5000
```

Experiment:

```
stock_movement_prediction
```

The best model is registered in MLflow Model Registry.

---

# Model Evaluation

Run:

```bash
python src/evaluate.py
```

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

# Continuous Integration with CML

GitHub Actions workflow:

```
.github/workflows/cml.yaml
```

The CI pipeline performs:

1. Checkout repository
2. Install dependencies
3. Pull dataset using DVC
4. Fetch MLflow registered model
5. Run evaluation
6. Generate CML report
7. Comment metrics on Pull Request

Workflow trigger:

```
main branch push
pull_request
```

---

# Google Cloud Platform Setup

The complete pipeline is executed on GCP.

Grant course team access:

```bash
PROJECT_ID=<YOUR_PROJECT_ID>

for ROLE in roles/viewer roles/container.viewer; do

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
--member="user:da5014_1@study.iitm.ac.in" \
--role="$ROLE"

done
```

Example:

```bash
PROJECT_ID=mlops-500009
```

---

# Results

## Iteration 1

Dataset:

```
v0
```

Model trained using only initial stock data.


## Iteration 2

Dataset:

```
v0 + v1
```

Model retrained with additional data.


Metrics are logged and compared using MLflow.

---

# Tools and Technologies

| Component | Technology |
|-|-|
| Programming Language | Python |
| ML Model | LightGBM |
| Data Processing | Pandas |
| Feature Store | Feast |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Storage | Google Cloud Storage |
| CI/CD | GitHub Actions |
| Reporting | CML |
| Cloud Platform | Google Cloud Platform |

---

# AI Usage Declaration

AI tools were used during development for:

- Debugging Python errors
- Understanding MLOps concepts
- Improving documentation
- Reviewing configuration files
- Suggesting implementation approaches

All implementation decisions, code execution, testing, debugging, and final integration were performed by the student.
