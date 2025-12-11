# Financial Stress Test Generator 🏦

[![Build Status](https://github.com/mlops-group-11/financial-stress-test/actions/workflows/retrain.yml/badge.svg)](https://github.com/mlops-group-11/financial-stress-test/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLOps](https://img.shields.io/badge/MLOps-Production-green.svg)](https://mlops.org/)
[![GCP](https://img.shields.io/badge/Cloud-GCP-4285F4.svg)](https://cloud.google.com/)

> **An end-to-end MLOps system for automated financial stress testing using machine learning.**

Financial Stress Test Generator is a production-grade machine learning pipeline that assesses how companies perform under adverse economic conditions. Built with modern MLOps practices, it automates everything from data ingestion to model deployment, enabling real-time risk assessment through an interactive dashboard.

## 🎯 Key Features

- **🎲 Synthetic Scenario Generation**: VAE model creates realistic economic stress scenarios
- **📊 Multi-Target Forecasting**: LightGBM ensemble predicts 5 financial metrics per company
- **⚠️ Intelligent Risk Assessment**: One-Class SVM with SHAP explainability for anomaly detection
- **🔄 Automated Pipeline**: Apache Airflow orchestrates daily data updates from FRED & Yahoo Finance
- **🚀 CI/CD Integration**: GitHub Actions automatically retrains models on code changes
- **☁️ Cloud-Native**: Deployed on GCP Cloud Run with auto-scaling
- **📈 Real-Time Dashboard**: Interactive UI for scenario generation and stress testing
- **📉 Drift Detection**: Automated monitoring for data and model performance drift
- **🔍 MLflow Tracking**: Complete experiment tracking and model versioning

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#-usage)
  - [Quick Start](#quick-start)
  - [API Usage](#api-usage)
  - [Dashboard Usage](#dashboard-usage)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Model Architecture](#-model-architecture)
- [Deployment](#-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Monitoring & Drift Detection](#-monitoring--drift-detection)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [Support](#-support)
- [Team](#-team)
- [License](#-license)

---

## 🏗 Architecture

Our system implements a complete MLOps pipeline with the following components:

```mermaid
graph LR
    %% Styles
    classDef data fill:#4A90E2,stroke:#357ABD,stroke-width:3px,color:#fff
    classDef process fill:#5CB85C,stroke:#449D44,stroke-width:3px,color:#fff
    classDef model fill:#F0AD4E,stroke:#EC971F,stroke-width:3px,color:#fff
    classDef storage fill:#34A853,stroke:#2E7D32,stroke-width:3px,color:#fff
    classDef cicd fill:#9B59B6,stroke:#8E44AD,stroke-width:3px,color:#fff
    classDef api fill:#E74C3C,stroke:#C9302C,stroke-width:3px,color:#fff
    classDef ui fill:#3498DB,stroke:#2980B9,stroke-width:3px,color:#fff
    
    Start([🌍 Data Sources<br/>FRED + Yahoo]):::data
    Pipeline[🔄 Airflow Pipeline<br/>Clean & Process]:::process
    Store1[(☁️ GCS<br/>Processed Data)]:::storage
    Train[🤖 Train 3 Models<br/>VAE + LightGBM + SVM]:::model
    Store2[(☁️ GCS<br/>Models .pkl)]:::storage
    API[⚡ FastAPI<br/>Serves Models]:::api
    UI[💼 Dashboard<br/>User Interface]:::ui
    Deploy([☁️ Cloud Run<br/>Deployed]):::cicd
    CICD[🔄 GitHub Actions<br/>Auto-Retrain]:::cicd
    MLflow[📊 MLflow<br/>Tracking]:::storage
    
    Start --> Pipeline --> Store1 --> Train --> Store2
    Store2 --> API --> UI --> Deploy
    CICD -.->|Triggers| Train
    CICD -.->|Uploads| Store2
    Store2 -.->|Auto-reload| API
    Train --> MLflow
```

### Pipeline Flow

1. **Data Ingestion**: FRED API and Yahoo Finance provide economic and financial data
2. **Data Processing**: Apache Airflow orchestrates cleaning and feature engineering
3. **Model Training**: Three specialized ML models train on processed data
4. **Model Storage**: Trained models (.pkl files) stored in Google Cloud Storage
5. **CI/CD Automation**: GitHub Actions triggers retraining on code changes
6. **API Serving**: FastAPI loads latest models and serves predictions
7. **User Interface**: React dashboard provides interactive stress testing
8. **Deployment**: Serverless deployment on GCP Cloud Run with auto-scaling

---

## 💻 Technology Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.9+ | Core development language |
| **ML Frameworks** | PyTorch, LightGBM, Scikit-learn | Model development |
| **Orchestration** | Apache Airflow | Workflow automation |
| **Experiment Tracking** | MLflow | Model versioning & metrics |
| **API Framework** | FastAPI | High-performance REST API |
| **Frontend** | React/HTML5 | Interactive dashboard |
| **Cloud Platform** | Google Cloud Platform | Infrastructure & deployment |
| **Containerization** | Docker | Application packaging |
| **CI/CD** | GitHub Actions | Automated testing & deployment |
| **Monitoring** | GCP Cloud Logging | System observability |

### Key Libraries

```
pytorch>=1.12.0
lightgbm>=3.3.5
scikit-learn>=1.1.0
snorkel>=0.9.9
fastapi>=0.95.0
apache-airflow>=2.5.0
mlflow>=2.3.0
pandas>=1.5.0
numpy>=1.23.0
```

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9 or higher** ([Download](https://www.python.org/downloads/))
- **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop))
- **Git** ([Download](https://git-scm.com/downloads))
- **Google Cloud SDK** ([Install Guide](https://cloud.google.com/sdk/docs/install))
- **GCP Account** with billing enabled

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/mlops-group-11/financial-stress-test.git
cd financial-stress-test
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

#### 4. Install Development Dependencies (Optional)

```bash
pip install -r requirements-dev.txt
```

### Configuration

#### 1. Set Up Google Cloud Credentials

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Set your GCP project
gcloud config set project YOUR_PROJECT_ID

# Create service account (if needed)
gcloud iam service-accounts create mlops-service-account \
    --display-name="MLOps Service Account"

# Download service account key
gcloud iam service-accounts keys create ~/gcp-key.json \
    --iam-account=mlops-service-account@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/gcp-key.json"
```

#### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# GCP Configuration
GCS_BUCKET_NAME=mlops-models
GCS_DATA_BUCKET=mlops-data
GCP_PROJECT_ID=your-project-id

# API Keys
FRED_API_KEY=your_fred_api_key_here

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

#### 3. Create GCS Buckets

```bash
# Create buckets for data and models
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://mlops-data
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://mlops-models

# Verify buckets
gsutil ls
```

#### 4. Get FRED API Key

1. Visit [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a free account
3. Request an API key
4. Add to `.env` file

---

## 📖 Usage

### Quick Start

#### Option 1: Run with Docker (Recommended)

```bash
# Build the Docker image
docker build -t financial-stress-test .

# Run the container
docker run -p 8000:8000 \
  --env-file .env \
  -v ~/.config/gcloud:/root/.config/gcloud \
  financial-stress-test

# Access the dashboard
open http://localhost:8000/dashboard
```

#### Option 2: Run Locally

```bash
# Step 1: Run data pipeline
python airflow/dags/data_pipeline.py

# Step 2: Train models
python models/model1_vae/train_vae.py
python models/model2_forecasting/train_lightgbm.py
python models/model3_anomaly/train_svm.py

# Step 3: Start API server
cd deployment/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Step 4: Open dashboard in browser
open http://localhost:8000/dashboard
```

### API Usage

#### Example 1: Check API Health

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_versions": {
    "model1": "2024-12-10T14:30:22Z",
    "model2": {
      "Revenue": "2024-12-10T14:35:10Z",
      "EPS": "2024-12-10T14:36:05Z"
    },
    "model3": "2024-12-10T14:40:00Z"
  }
}
```

#### Example 2: Generate Stress Scenarios

```bash
curl -X POST http://localhost:8000/api/v1/scenarios/generate \
  -H "Content-Type: application/json" \
  -d '{"n_scenarios": 20}'
```

**Response:**
```json
{
  "status": "success",
  "n_scenarios": 20,
  "scenarios": [
    {
      "scenario_id": 1,
      "severity": "adverse",
      "crisis_type": "market_crash",
      "macroeconomic_indicators": {
        "GDP": 24500.0,
        "VIX": 28.5,
        "Unemployment_Rate": 5.2,
        "CPI": 315.2,
        "Federal_Funds_Rate": 6.0
      }
    }
  ]
}
```

#### Example 3: Run Stress Test

```bash
curl -X POST http://localhost:8000/api/v1/stress-test \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "AAPL",
    "scenario_ids": [1, 2, 3]
  }'
```

**Response:**
```json
{
  "company_id": "AAPL",
  "n_scenarios": 3,
  "aggregated": true,
  "summary": {
    "avg_risk_score": 42.5,
    "best_case": {
      "scenario_id": 1,
      "risk_score": 25.3,
      "predictions": {
        "revenue_change_pct": -5.2,
        "eps_change_pct": -8.1
      }
    },
    "worst_case": {
      "scenario_id": 3,
      "risk_score": 68.2,
      "predictions": {
        "revenue_change_pct": -18.5,
        "eps_change_pct": -25.3
      }
    }
  }
}
```

#### Example 4: Python Client

```python
import requests

# Initialize client
API_BASE = "http://localhost:8000/api/v1"

# Generate scenarios
response = requests.post(
    f"{API_BASE}/scenarios/generate",
    json={"n_scenarios": 50}
)
scenarios = response.json()

# Run stress test
response = requests.post(
    f"{API_BASE}/stress-test",
    json={
        "company_id": "AAPL",
        "scenario_ids": [1, 2, 3, 4, 5]
    }
)
results = response.json()

print(f"Average Risk Score: {results['summary']['avg_risk_score']}")
```

### Dashboard Usage

#### Step 1: Generate Scenarios
1. Open the dashboard at `http://localhost:8000/dashboard`
2. Navigate to the **"Generate Scenarios"** tab
3. Select the number of scenarios (10, 20, 50, or 100)
4. Click **"Generate New Scenarios"**
5. Review generated scenarios and click on cards to select them (selected cards turn green)

#### Step 2: Run Stress Test
1. Navigate to the **"Stress Test"** tab
2. Select a company from the dropdown (e.g., AAPL, MSFT, GOOGL)
3. Choose **"Use Selected Scenarios from Tab 2"**
4. Click **"Run Stress Test"**
5. Results automatically appear in the **"Results"** tab

#### Step 3: Analyze Results
View comprehensive results including:
- **Risk Score**: 0-100 scale with risk category (Low/Moderate/High/Critical)
- **Economic Scenario**: GDP, VIX, unemployment, and other indicators
- **Predicted Financials**: Revenue, EPS, Debt-to-Equity, Profit Margin, ROA
- **SHAP Explanations**: Top 5 risk factors with contribution scores
- **Anomaly Detection**: Whether the prediction is anomalous

#### Step 4: Portfolio Analysis (Advanced)
1. Navigate to the **"Portfolio Analysis"** tab
2. Add multiple companies with weights (must sum to 100%)
3. Select scenarios to test
4. Click **"Analyze Portfolio"**
5. View weighted risk scores and sector breakdown

---

## 📁 Project Structure

```
financial-stress-test-mlops/
│
├── .github/
│   └── workflows/
│       ├── retrain.yml              # CI/CD pipeline for model retraining
│       └── tests.yml                # Automated testing workflow
│
├── data/
│   ├── raw/                         # Raw data from FRED & Yahoo Finance
│   ├── processed/                   # Cleaned and feature-engineered data
│   └── scenarios/                   # Generated stress scenarios
│
├── models/
│   ├── model1_vae/
│   │   ├── train_vae.py            # VAE training script
│   │   ├── vae_model.py            # VAE architecture
│   │   ├── generate_scenarios.py   # Scenario generation logic
│   │   └── config.yaml             # Model hyperparameters
│   │
│   ├── model2_forecasting/
│   │   ├── train_lightgbm.py       # LightGBM training for 5 targets
│   │   ├── feature_engineering.py  # Feature preprocessing
│   │   ├── evaluate.py             # Model evaluation metrics
│   │   └── config.yaml             # Hyperparameters per target
│   │
│   └── model3_anomaly/
│       ├── train_svm.py            # One-Class SVM training
│       ├── snorkel_labeling.py     # Weak supervision with Snorkel
│       ├── explain_shap.py         # SHAP-based explanations
│       └── config.yaml             # Model configuration
│
├── deployment/
│   ├── api/
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── config.py               # API configuration
│   │   ├── model_loader.py         # GCS model loader with caching
│   │   ├── endpoints/
│   │   │   ├── health.py           # Health check endpoint
│   │   │   ├── scenarios.py        # Scenario generation endpoints
│   │   │   ├── stress_test.py      # Stress testing endpoints
│   │   │   └── companies.py        # Company listing endpoint
│   │   └── utils/
│   │       ├── preprocessing.py    # Input preprocessing
│   │       └── validators.py       # Request validation
│   │
│   └── frontend/
│       ├── dashboard.html          # Main dashboard interface
│       ├── styles.css              # Dashboard styling
│       └── app.js                  # Dashboard logic
│
├── airflow/
│   ├── dags/
│   │   ├── data_pipeline.py        # Main data pipeline DAG
│   │   ├── fred_ingestion.py       # FRED API data fetching
│   │   └── yahoo_ingestion.py      # Yahoo Finance data fetching
│   │
│   └── config/
│       └── airflow.cfg             # Airflow configuration
│
├── monitoring/
│   ├── drift_detection.py          # Data drift monitoring
│   ├── performance_tracking.py     # Model performance metrics
│   └── alerts.py                   # Alerting system
│
├── tests/
│   ├── unit/
│   │   ├── test_models.py          # Unit tests for models
│   │   ├── test_preprocessing.py   # Data processing tests
│   │   └── test_utils.py           # Utility function tests
│   │
│   └── integration/
│       ├── test_api.py             # API integration tests
│       └── test_pipeline.py        # End-to-end pipeline tests
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA and data analysis
│   ├── 02_model_experiments.ipynb  # Model experimentation
│   └── 03_results_analysis.ipynb   # Results visualization
│
├── docs/
│   ├── API.md                      # Detailed API documentation
│   ├── DEPLOYMENT.md               # Deployment guide
│   ├── CONTRIBUTING.md             # Contribution guidelines
│   └── ARCHITECTURE.md             # System architecture details
│
├── scripts/
│   ├── setup_gcp.sh                # GCP infrastructure setup
│   ├── download_data.sh            # Data download script
│   └── deploy.sh                   # Deployment automation
│
├── Dockerfile                       # Docker image definition
├── docker-compose.yml              # Multi-container Docker setup
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── .env.example                    # Example environment variables
├── .gitignore                      # Git ignore rules
├── .dockerignore                   # Docker ignore rules
├── README.md                       # This file
└── LICENSE                         # MIT License
```

---

## 🔄 Data Pipeline

### Data Sources

#### 1. FRED API (Economic Data)
**Indicators Collected:**
- GDP (Gross Domestic Product)
- CPI (Consumer Price Index)
- Unemployment Rate
- Federal Funds Rate
- VIX (Market Volatility Index)
- 10-Year Treasury Yield

**Update Frequency:** Daily/Weekly/Monthly

#### 2. Yahoo Finance API (Financial Data)
**Company Metrics:**
- Stock prices (OHLC)
- Revenue
- Earnings Per Share (EPS)
- Debt-to-Equity Ratio
- Profit Margin
- Return on Assets (ROA)

**Companies Tracked:** 50+ major US companies

### Pipeline Workflow

```
┌─────────────────┐
│  FRED API       │─┐
│  Yahoo Finance  │ │
└─────────────────┘ │
                    ▼
┌──────────────────────────┐
│  Apache Airflow DAG      │
│  ┌────────────────────┐  │
│  │ 1. Fetch Data      │  │
│  │ 2. Validate Schema │  │
│  │ 3. Clean & Process │  │
│  │ 4. Feature Eng.    │  │
│  │ 5. Upload to GCS   │  │
│  └────────────────────┘  │
└──────────────────────────┘
                    │
                    ▼
┌──────────────────────────┐
│  GCS Storage             │
│  gs://mlops-data/        │
│    ├── raw/              │
│    └── processed/        │
└──────────────────────────┘
```

### Data Processing Steps

1. **Ingestion**: Fetch latest data from APIs
2. **Validation**: Check data quality and schema
3. **Cleaning**: Handle missing values, remove outliers
4. **Feature Engineering**: Create derived features, calculate ratios
5. **Storage**: Upload to GCS with timestamps

### Running the Pipeline

```bash
# Manually trigger pipeline
python airflow/dags/data_pipeline.py

# Or use Airflow CLI
airflow dags trigger financial_data_pipeline
```

---

## 🤖 Model Architecture

### Overview

Our system uses three specialized models working in ensemble:

| Model | Purpose | Technology | Output |
|-------|---------|------------|--------|
| **Model 1** | Scenario Generation | Variational Autoencoder (VAE) | Synthetic economic scenarios |
| **Model 2** | Financial Forecasting | LightGBM Ensemble | 5 financial metrics predictions |
| **Model 3** | Risk Assessment | One-Class SVM + Snorkel | Anomaly score & risk category |

### Model 1: VAE for Scenario Generation

**Architecture:**
```
Input (15 features) → Encoder [64, 32] → Latent (16) → Decoder [32, 64] → Output (15 features)
```

**Key Features:**
- Generates synthetic economic stress scenarios
- Latent space sampling for scenario diversity
- Severity classification (Baseline/Adverse/Severe/Extreme)

**Training:**
```bash
python models/model1_vae/train_vae.py \
    --epochs 100 \
    --latent_dim 16 \
    --learning_rate 0.001
```

**Output:** `model1_vae.pkl` saved to GCS

### Model 2: LightGBM Forecasting Ensemble

**Target Variables:**
1. Revenue
2. Earnings Per Share (EPS)
3. Debt-to-Equity Ratio
4. Profit Margin
5. Return on Assets (ROA)

**Architecture:** 5 independent LightGBM models (one per target)

**Hyperparameters:**
```python
{
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "max_depth": 7,
    "min_child_samples": 20,
    "objective": "regression"
}
```

**Training:**
```bash
python models/model2_forecasting/train_lightgbm.py \
    --targets Revenue,EPS,Debt_to_Equity,Profit_Margin,ROA
```

**Output:** 5 `.pkl` files (one per target) saved to GCS

### Model 3: One-Class SVM for Anomaly Detection

**Architecture:**
1. **Snorkel Weak Supervision**: Label unlabeled data using labeling functions
2. **One-Class SVM**: Train on "normal" financial outcomes
3. **SHAP Explainer**: Generate feature importance for interpretability

**Workflow:**
```
Predictions → Snorkel Labeling → SVM Training → Anomaly Score → SHAP Explanations
```

**Training:**
```bash
python models/model3_anomaly/train_svm.py \
    --kernel rbf \
    --gamma scale \
    --nu 0.1
```

**Output:** `model3_svm.pkl` with scaler and feature list saved to GCS

### Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| VAE | Reconstruction Loss | 0.032 |
| LightGBM (Revenue) | RMSE | $2.5B |
| LightGBM (EPS) | MAE | $0.15 |
| One-Class SVM | ROC-AUC | 0.89 |

---

## ☁️ Deployment

### Docker Deployment

#### Build Image

```bash
# Build Docker image
docker build -t financial-stress-test:latest .

# Tag for GCR
docker tag financial-stress-test:latest \
    gcr.io/YOUR_PROJECT_ID/financial-stress-test:latest

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/financial-stress-test:latest
```

#### Run Locally

```bash
docker run -p 8000:8000 \
    --env-file .env \
    -v ~/.config/gcloud:/root/.config/gcloud \
    financial-stress-test:latest
```

### GCP Cloud Run Deployment

#### Deploy Command

```bash
gcloud run deploy financial-stress-test-api \
    --image gcr.io/YOUR_PROJECT_ID/financial-stress-test:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --set-env-vars GCS_BUCKET_NAME=mlops-models,GCS_DATA_BUCKET=mlops-data
```

#### Configuration

- **Memory:** 4GB
- **CPU:** 2 vCPUs
- **Min Instances:** 1 (always warm)
- **Max Instances:** 10 (auto-scaling)
- **Timeout:** 300 seconds
- **Concurrency:** 100 requests per instance

#### Access Deployed API

```bash
# Get service URL
gcloud run services describe financial-stress-test-api \
    --region us-central1 \
    --format 'value(status.url)'

# Test health endpoint
curl $(gcloud run services describe financial-stress-test-api \
    --region us-central1 \
    --format 'value(status.url)')/api/v1/health
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### 1. Model Retraining Pipeline (`.github/workflows/retrain.yml`)

**Triggers:**
- Push to `main` branch in `models/` or `data/` directories
- Weekly schedule (every Sunday at midnight)
- Manual workflow dispatch

**Steps:**
1. Checkout code
2. Set up Python 3.9
3. Install dependencies
4. Authenticate to GCP
5. Download latest data from GCS
6. Train Model 1 (VAE)
7. Train Model 2 (LightGBM)
8. Train Model 3 (SVM)
9. Upload trained models to GCS
10. Trigger API reload endpoint
11. Run integration tests

**Workflow:**
```yaml
name: Model Retraining Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'data/**'
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Train models
        run: |
          python models/model1_vae/train_vae.py
          python models/model2_forecasting/train_lightgbm.py
          python models/model3_anomaly/train_svm.py
      - name: Upload to GCS
        run: |
          gsutil cp models/*.pkl gs://mlops-models/
      - name: Reload API
        run: |
          curl -X POST ${{ secrets.API_URL }}/api/v1/admin/reload-models
```

#### 2. Testing Pipeline (`.github/workflows/tests.yml`)

**Triggers:**
- Pull request to `main` branch
- Push to any branch

**Steps:**
1. Run unit tests
2. Run integration tests
3. Check code coverage
4. Lint code (flake8, black)
5. Type checking (mypy)

### Model Versioning Strategy

- **Naming Convention:** `model_YYYYMMDD_HHMMSS.pkl`
- **Storage:** GCS with automatic versioning enabled
- **Rollback:** Previous versions retained for 30 days
- **Loading:** API always loads latest model by timestamp

---

## 📊 Monitoring & Drift Detection

### Data Drift Detection

**Method:** Kolmogorov-Smirnov (KS) Statistical Test

**Monitored Features:**
- Economic indicators (GDP, VIX, unemployment)
- Company financials (revenue, EPS, debt ratios)

**Detection Logic:**
```python
from scipy.stats import ks_2samp

def detect_drift(reference_data, current_data, threshold=0.05):
    """
    Detect distribution drift using KS test
    
    Returns:
        True if drift detected (p-value < threshold)
    """
    statistic, p_value = ks_2samp(reference_data, current_data)
    return p_value < threshold
```

**Alert Threshold:** p-value < 0.05

### Model Performance Monitoring

**Metrics Tracked:**
- Prediction RMSE over time
- Error distribution shifts
- Inference latency (p50, p95, p99)
- API response times

**Dashboard:** Real-time metrics in GCP Cloud Logging

### Alerting System

**Alert Conditions:**
- Data drift detected in >3 features
- Model RMSE increases >10%
- API error rate exceeds 5%
- Inference latency >2 seconds

**Notification Channels:**
- Email alerts to team
- Slack webhooks (#mlops-alerts)
- GCP Cloud Monitoring

**Setup Alerts:**
```bash
# Create alert policy in GCP
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="Model Performance Alert" \
    --condition-display-name="High Error Rate" \
    --condition-threshold-value=0.05 \
    --condition-threshold-duration=300s
```

---

## 📚 API Documentation

### Base URL

- **Local:** `http://localhost:8000`
- **Production:** `https://financial-stress-test-api-[hash]-uc.a.run.app`

### Authentication

Currently no authentication required. For production, consider adding:
- API keys
- OAuth 2.0
- JWT tokens

### Endpoints

#### GET `/api/v1/health`

Check API health status and model versions.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_versions": {
    "model1": "2024-12-10T14:30:22Z",
    "model2": {
      "Revenue": "2024-12-10T14:35:10Z",
      "EPS": "2024-12-10T14:36:05Z"
    },
    "model3": "2024-12-10T14:40:00Z"
  },
  "n_companies": 50,
  "n_scenarios": 100
}
```

#### GET `/api/v1/companies`

List all available companies.

**Response:**
```json
{
  "companies": [
    {
      "company_id": "AAPL",
      "name": "Apple Inc.",
      "sector": "Technology",
      "market_cap": 3000000000000
    }
  ]
}
```

#### GET `/api/v1/scenarios`

List all generated scenarios.

**Response:**
```json
{
  "scenarios": [
    {
      "scenario_id": 1,
      "severity": "adverse",
      "crisis_type": "market_crash",
      "preview": {
        "GDP": 24500.0,
        "VIX": 28.5,
        "Unemployment_Rate": 5.2
      }
    }
  ]
}
```

#### POST `/api/v1/scenarios/generate`

Generate new stress scenarios.

**Request Body:**
```json
{
  "n_scenarios": 50
}
```

**Response:**
```json
{
  "status": "success",
  "n_scenarios": 50,
  "scenarios": [...]
}
```

#### POST `/api/v1/stress-test`

Run stress test for a company.

**Request Body:**
```json
{
  "company_id": "AAPL",
  "scenario_ids": [1, 2, 3]
}
```

**Response:**
```json
{
  "company_id": "AAPL",
  "n_scenarios": 3,
  "aggregated": true,
  "summary": {
    "avg_risk_score": 42.5,
    "best_case": {...},
    "worst_case": {...}
  },
  "detailed_results": [...]
}
```

#### POST `/api/v1/admin/reload-models`

Force reload models from GCS (admin only).

**Query Parameters:**
- `force` (boolean): Ignore cache and re-download

**Response:**
```json
{
  "status": "success",
  "message": "Models reloaded successfully",
  "loaded_at": "2024-12-10T15:00:00Z"
}
```

### Error Responses

All endpoints return consistent error format:

```json
{
  "detail": "Error message here",
  "status_code": 400,
  "timestamp": "2024-12-10T15:00:00Z"
}
```

**Common Status Codes:**
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error
- `503`: Service Unavailable

---

## 🤝 Contributing

We welcome contributions from the community! Please follow these guidelines:

### How to Contribute

1. **Fork the Repository**
   ```bash
   # Fork via GitHub UI, then clone
   git clone https://github.com/YOUR_USERNAME/financial-stress-test.git
   cd financial-stress-test
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests for new features
   - Update documentation as needed

4. **Run Tests**
   ```bash
   # Run all tests
   pytest tests/

   # Check code style
   flake8 .
   black --check .

   # Type checking
   mypy models/ deployment/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   **Commit Message Format:**
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `chore:` Maintenance tasks

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template

### Code Style Guidelines

- **Python:** Follow PEP 8
- **Line Length:** Max 88 characters (Black default)
- **Docstrings:** Use Google-style docstrings
- **Type Hints:** Add type hints to all functions
- **Comments:** Explain complex logic

### Testing Requirements

- All new features must include unit tests
- Maintain >80% code coverage
- Integration tests for API endpoints
- Test edge cases and error handling

### Review Process

1. Automated checks must pass (GitHub Actions)
2. Code review by at least one maintainer
3. All comments must be addressed
4. Squash commits before merging

For detailed guidelines, see [CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

## 💬 Support

### Getting Help

- **📖 Documentation:** Check our [docs](docs/) folder for detailed guides
- **🐛 Bug Reports:** Open an issue on [GitHub Issues](https://github.com/mlops-group-11/financial-stress-test/issues)
- **💡 Feature Requests:** Use the feature request template
- **❓ Questions:** Ask in [Discussions](https://github.com/mlops-group-11/financial-stress-test/discussions)

### Contact

- **Email:** mlops-group-11@northeastern.edu

### FAQ

**Q: How often are models retrained?**  
A: Automatically every Sunday via GitHub Actions, or when changes are pushed to model code.

**Q: Can I use this with my own data?**  
A: Yes! Replace the data ingestion logic in `airflow/dags/` with your data sources.

**Q: What's the API rate limit?**  
A: Currently no limits. For production, we recommend implementing rate limiting.

**Q: How do I add a new company?**  
A: Add the company ticker to `data/company_list.csv` and re-run the data pipeline.

---

## 👥 Team

**MLOps Group 11 - Northeastern University**

| Member | Role | Contact |
|--------|------|---------|
| **Sushmitha** | Model Development | [@sushmitha](https://github.com/sushmitha) |
| **Priyanka Senthil Kumar** | Data Engineering, Pipeline | [@member2](https://github.com/member2) |
| **Parth** | API Development, Deployment | [@member3](https://github.com/member3) |
| **Sanika** | Frontend, Documentation | [@member4](https://github.com/member4) |
| **Sailee** | Frontend, Documentation | [@member4](https://github.com/member4) |
| **Novia** | Frontend, Documentation | [@member4](https://github.com/member4) |

### Acknowledgments

- **Course Instructor:** Prof. Ramin Mohammadi, Northeastern University
- **Data Providers:** Federal Reserve Economic Data (FRED), Yahoo Finance, Alpha Vantage
- **Cloud Provider:** Google Cloud Platform

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 MLOps Group 11, Northeastern University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text in LICENSE file]
```

---

## 📊 Project Status

![Status](https://img.shields.io/badge/status-active-success.svg)
![Development](https://img.shields.io/badge/development-in--progress-blue.svg)

**Current Version:** 1.0.0  
**Last Updated:** December 10, 2024  
**Maintenance:** Actively maintained

### Roadmap

- [x] MVP: Basic stress testing functionality
- [x] Model training pipeline
- [x] CI/CD automation
- [x] Cloud deployment

---

## 🔗 Additional Resources

- **Course Website:** [MLOps Fall 2024](https://northeastern.edu/mlops)
- **FRED API Docs:** https://fred.stlouisfed.org/docs/api/
- **Yahoo Finance API:** https://finance.yahoo.com/
- **MLflow Documentation:** https://mlflow.org/docs/latest/
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **GCP Cloud Run:** https://cloud.google.com/run/docs

---

**Built by MLOps Group 11 at Northeastern University**

*This project is submitted as part of the MLOps course (Fall 2024)*
