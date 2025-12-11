# Financial Stress Test Generator

## Introduction

The **Financial Stress Test Generator** is an end-to-end MLOps system designed to predict company financial distress under extreme economic conditions. Deployed on **Google Cloud Platform** with automated **CI/CD pipelines**, the system generates synthetic economic stress scenarios, forecasts company-specific financial outcomes, and identifies at-risk entities through advanced anomaly detection.

The system operates through **three specialized machine learning models** working in sequence:

**Model 1 - VAE Scenario Generator**: A Variational Autoencoder generates **100 synthetic economic stress scenarios** across four severity levels—Baseline, Adverse, Severe, and Extreme—trained on **35 years of historical crisis data** including the 2008 financial crisis, COVID-19 pandemic, and dot-com bubble.

**Model 2 - Financial Forecasting Ensemble**: XGBoost and LightGBM models predict **five critical financial metrics** for each company: Revenue, EPS (Earnings Per Share), Debt-to-Equity Ratio, Profit Margin, and Stock Returns. Each target variable independently selects the **best-performing algorithm** through rigorous validation.

**Model 3 - Anomaly Detection with Snorkel**: A One-Class SVM with **Snorkel weak supervision** analyzes predictions to generate a numerical stress score indicating overall distress likelihood, **completely eliminating manual labeling** and saving **$2,000-$4,000 per assessment**.

The infrastructure leverages **Google Cloud Storage** for all data and model artifacts, **Apache Airflow** for daily orchestration, **MLflow** for experiment tracking across **40+ training runs**, and **DVC** for complete data versioning. The **CI/CD pipeline** via GitHub Actions implements **intelligent selective retraining**—only affected models retrain when code changes occur. The system incorporates **comprehensive bias detection** across three dimensions: economic condition bias, crisis-based bias, and sector-based bias, ensuring fair and accurate predictions across all market conditions and industries.

## System Architecture

flowchart TB
    %% Simplified, Clean Architecture Diagram
    
    %% Style Definitions
    classDef dataSource fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef storage fill:#34A853,stroke:#1E7E34,stroke-width:2px,color:#fff
    classDef model fill:#FBBC04,stroke:#E37400,stroke-width:2px,color:#000
    classDef cicd fill:#9B59B6,stroke:#6C3483,stroke-width:2px,color:#fff
    classDef api fill:#E74C3C,stroke:#C0392B,stroke-width:2px,color:#fff
    classDef ui fill:#3498DB,stroke:#21618C,stroke-width:2px,color:#fff
    
    %% ====================
    %% LAYER 1: DATA SOURCES
    %% ====================
    subgraph Layer1["📥 DATA SOURCES"]
        direction LR
        FRED[FRED API<br/>Economic Data]:::dataSource
        Yahoo[Yahoo Finance<br/>Market Data]:::dataSource
    end
    
    %% ====================
    %% LAYER 2: DATA PIPELINE
    %% ====================
    subgraph Layer2["🔄 DATA PIPELINE"]
        Airflow[Apache Airflow<br/>Orchestration]:::dataSource
        GCS1[(Google Cloud Storage<br/>Raw & Processed Data)]:::storage
    end
    
    %% ====================
    %% LAYER 3: MODEL TRAINING
    %% ====================
    subgraph Layer3["🤖 MODEL TRAINING"]
        direction LR
        M1[Model 1: VAE<br/>Scenario Generation]:::model
        M2[Model 2: LightGBM<br/>Financial Forecasting]:::model
        M3[Model 3: SVM<br/>Anomaly Detection]:::model
    end
    
    %% ====================
    %% LAYER 4: MODEL STORAGE
    %% ====================
    GCS2[(GCS Bucket<br/>Model Artifacts<br/>.pkl files)]:::storage
    MLflow[MLflow<br/>Experiment Tracking]:::storage
    
    %% ====================
    %% LAYER 5: CI/CD
    %% ====================
    subgraph Layer5["🔄 CI/CD AUTOMATION"]
        GitHub[GitHub Actions<br/>Auto-Retrain on Push]:::cicd
    end
    
    %% ====================
    %% LAYER 6: API BACKEND
    %% ====================
    subgraph Layer6["⚡ API BACKEND"]
        FastAPI[FastAPI Server<br/>+ Model Loader<br/>Auto-reload from GCS]:::api
    end
    
    %% ====================
    %% LAYER 7: FRONTEND
    %% ====================
    subgraph Layer7["💼 USER INTERFACE"]
        Dashboard[React Dashboard<br/>Generate Scenarios<br/>Run Stress Tests<br/>View Results]:::ui
    end
    
    %% ====================
    %% LAYER 8: DEPLOYMENT
    %% ====================
    Deploy[GCP Cloud Run<br/>Serverless Deployment]:::cicd
    
    %% ====================
    %% CONNECTIONS
    %% ====================
    
    %% Data Flow
    FRED --> Airflow
    Yahoo --> Airflow
    Airflow --> GCS1
    GCS1 --> M1 & M2 & M3
    
    %% Model Training Flow
    M1 & M2 & M3 --> MLflow
    M1 & M2 & M3 --> GCS2
    
    %% CI/CD Flow
    GitHub -.->|Trigger<br/>Retrain| M1 & M2 & M3
    GitHub -.->|Upload New<br/>Models| GCS2
    
    %% API Flow
    GCS2 -->|Fetch Latest<br/>Models| FastAPI
    FastAPI <-->|API Calls| Dashboard
    
    %% Deployment
    FastAPI --> Deploy
    
    %% ====================
    %% LABELS
    %% ====================
    
    style Layer1 fill:#E8F4F8,stroke:#4A90E2,stroke-width:3px
    style Layer2 fill:#E8F5E9,stroke:#5CB85C,stroke-width:3px
    style Layer3 fill:#FFF3E0,stroke:#F0AD4E,stroke-width:3px
    style Layer5 fill:#F5E6FF,stroke:#9B59B6,stroke-width:3px
    style Layer6 fill:#FCE4EC,stroke:#E91E63,stroke-width:3px
    style Layer7 fill:#E3F2FD,stroke:#2196F3,stroke-width:3px

The architecture illustrates the complete workflow from data ingestion through Apache Airflow, independent preprocessing pipelines for each model, parallel training with MLflow tracking, automated deployment to Google Cloud Storage, and intelligent CI/CD via GitHub Actions.

## Cloud Infrastructure

**Cloud Provider**: Google Cloud Platform (GCP)

**Core Services**:
- **Google Cloud Storage**: Data lake and model artifacts
- **Apache Airflow**: Workflow orchestration and scheduling
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data versioning with GCS remote storage
- **Docker**: Containerization for consistent deployment
- **FastAPI**: Backend API services
- **GitHub Actions**: CI/CD automation

## Key Features

- **Daily Automated Data Ingestion** with comprehensive quality validation via Great Expectations
- **596 Engineered Features** from 97 base features with point-in-time correctness
- **Three Independent Model Pipelines** with specialized preprocessing tailored to each model
- **VAE Scenario Generation**: 100 diverse scenarios with **80.6% statistical validity**
- **Ensemble Forecasting**: Achieves **0.78 average R²** across five financial targets
- **Zero Labeling Cost**: Snorkel weak supervision eliminates **$2K-$4K** manual labeling per assessment
- **Anomaly Detection**: One-Class SVM reaches **0.82 ROC-AUC** (96% of target)
- **Complete Reproducibility**: MLflow tracking across **40+ experiments** with DVC versioning
- **Three-Tier Bias Detection**: Economic, crisis-based, and sector fairness monitoring
- **Intelligent CI/CD**: Selective model retraining saves compute costs
- **90-Minute Pipeline**: Complete end-to-end execution with **zero manual intervention**

## Documentation

Comprehensive guides are available for all aspects of the project:

- [Project Setup Guide](readme/SETUP.md)
- [Data Pipeline Architecture](readme/DATA_PIPELINE.md)
- [Feature Engineering Guide](readme/FEATURE_ENGINEERING.md)
- [Model 1: VAE Scenario Generation](readme/VAE_SCENARIO_GENERATION.md)
- [Model 2: Financial Forecasting](readme/FINANCIAL_FORECASTING.md)
- [Model 3: Anomaly Detection with Snorkel](readme/ANOMALY_DETECTION.md)
- [MLflow Experiment Tracking](readme/MLFLOW_TRACKING.md)
- [DVC Data Versioning](readme/DVC_USAGE.md)
- [Bias Detection Framework](readme/BIAS_DETECTION.md)
- [CI/CD Workflow](readme/CICD_WORKFLOW.md)
- [GCP Deployment Guide](readme/DEPLOYMENT.md)
- [Business Value & ROI Analysis](readme/BUSINESS_VALUE.md)
- [Monitoring and Validation](readme/MONITORING.md)
- [Troubleshooting Guide](readme/TROUBLESHOOTING.md)

## Installation

### Prerequisites

Ensure you have the following installed:
- **Python 3.11+**
- **Docker** and **Docker Compose**
- **Git**
- **Google Cloud SDK** (for GCP deployment)

### Clone the Repository

```bash
git clone <repository-url>
cd financial-stress-test
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
- `FRED_API_KEY` - Get from https://fred.stlouisfed.org/
- `ALPHA_VANTAGE_API_KEY` - Get from https://www.alphavantage.co/
- `GCP_PROJECT_ID` - Your Google Cloud project ID
- `GCS_BUCKET_NAME` - Your GCS bucket name

### Set Up Google Cloud Storage

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create GCS bucket for data storage
gsutil mb gs://financial-stress-data

# Create GCS bucket for model artifacts
gsutil mb gs://financial-stress-models
```

### Initialize Airflow

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

### Start Services

```bash
# Option 1: Using Docker Compose (Recommended)
docker-compose up -d

# Option 2: Manual startup
airflow webserver --port 8080 &
airflow scheduler &
mlflow ui --port 5000 &
```

### Initialize DVC

```bash
dvc init
dvc remote add -d gcs gs://financial-stress-data/dvc-storage
dvc remote modify gcs projectname YOUR_PROJECT_ID
```

### Access Services

- **Airflow UI**: http://localhost:8080 (admin / admin)
- **MLflow UI**: http://localhost:5000

## Usage

### Running the Complete Pipeline

#### Step 1: Data Pipeline

Access Airflow UI and trigger the `financial_data_pipeline` DAG:

```bash
# Or trigger via CLI
airflow dags trigger financial_data_pipeline
```

The pipeline will:
- Fetch data from FRED API and Yahoo Finance
- Validate data quality with Great Expectations
- Process and engineer 596 features
- Store in Google Cloud Storage
- Create DVC version snapshot

#### Step 2: Train Models

```bash
# Generate stress scenarios (~10 minutes)
python src/scenario_generation/Dense_VAE_optimized_mlflow_updated.py

# Prepare forecasting data (~5 minutes)
python src/preprocessing/create_targets.py
python src/preprocessing/drop_leakage_features.py
python src/preprocessing/temporal_split.py
python src/preprocessing/handle_outliers_after_split.py

# Train forecasting models (~60 minutes)
python src/models/xgboost_model.py --target all
python src/models/lightgbm_model.py --target all
python src/models/model_selection.py

# Train anomaly detection (~10 minutes)
python src/eda/eda.py
python src/labeling/auto_threshold_extractor.py
python src/labeling/snorkel_pipeline.py
python src/models/train_anomaly_detection.py

# Run bias detection (~5 minutes)
python src/evaluation/test_all_models_for_bias.py --target all
python src/models/final_selection_after_bias_detection.py
```

**Total Runtime**: ~90 minutes with zero manual intervention

#### Step 3: View Results

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open browser to http://localhost:5000
```

### Production Inference

```python
import pickle
import torch
import numpy as np

# 1. Generate Stress Scenario
vae = torch.load('outputs/dense_vae_optimized_model.pth')
z = torch.randn(1, 32) * 2.5  # Severe scenario
scenario = vae.decoder(z).detach().numpy()

# 2. Predict Financial Outcomes
models = {
    'revenue': pickle.load(open('models/lightgbm_tuned/lightgbm_revenue_tuned.pkl', 'rb')),
    'eps': pickle.load(open('models/lightgbm_tuned/lightgbm_eps_tuned.pkl', 'rb')),
    'debt_equity': pickle.load(open('models/lightgbm_tuned/lightgbm_debt_equity_tuned.pkl', 'rb')),
    'profit_margin': pickle.load(open('models/xgboost_tuned/xgboost_profit_margin_tuned.pkl', 'rb')),
    'stock_return': pickle.load(open('models/lightgbm_tuned/lightgbm_stock_return_tuned.pkl', 'rb'))
}

predictions = {target: model.predict(scenario)[0] for target, model in models.items()}

# 3. Assess Risk
risk_model = pickle.load(open('models/anomaly_detection/One_Class_SVM/model.pkl', 'rb'))
scaler = pickle.load(open('models/anomaly_detection/One_Class_SVM/scaler.pkl', 'rb'))

X_scaled = scaler.transform(scenario)
risk_prediction = risk_model.predict(X_scaled)  # -1 = AT_RISK, 1 = NORMAL
risk_score = -risk_model.score_samples(X_scaled)

# Normalize to 0-100 scale
risk_score_normalized = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min()) * 100

print(f"Predicted Revenue: ${predictions['revenue']:,.0f}")
print(f"Predicted EPS: ${predictions['eps']:.2f}")
print(f"Risk Assessment: {'AT_RISK' if risk_prediction[0] == -1 else 'NORMAL'}")
print(f"Risk Score: {risk_score_normalized[0]:.1f}/100")
```

## Project Structure

```
financial-stress-test/
├── configs/                   # Configuration files (YAML)
│   ├── eda_config.yaml
│   ├── model_config.yaml
│   └── best_hyperparameters.yaml
├── data/                      # Data storage (gitignored)
│   ├── features/             # Feature-engineered datasets
│   └── splits/               # Train/val/test splits
├── src/
│   ├── scenario_generation/  # VAE models
│   ├── preprocessing/        # Data preparation scripts
│   ├── models/               # Model training scripts
│   ├── eda/                  # Exploratory data analysis
│   ├── labeling/             # Snorkel weak supervision
│   └── evaluation/           # Bias detection and validation
├── pipelines/
│   └── airflow/              # Airflow DAG definitions
├── outputs/                   # Model outputs and reports
├── models/                    # Trained model artifacts (.pkl, .pth)
├── reports/                   # Validation and bias reports
├── mlruns/                    # MLflow tracking data
├── logs/                      # Execution logs
├── docker-compose.yml         # Service orchestration
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Data Versioning

This project uses **DVC** (Data Version Control) to manage and version datasets:

```bash
# Pull latest data
dvc pull

# After processing new data
dvc add data/raw data/processed
git add data/raw.dvc data/processed.dvc
git commit -m "Update data version YYYY-MM-DD"
dvc push
```

Complete lineage is maintained from raw data through processed features to trained models.

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

**Current Coverage**: 84%

## Model Performance Summary

### VAE Scenario Generator
- **KS Pass Rate**: 80.6% (58/72 features pass statistical validity)
- **Correlation MAE**: 0.0645 (preserves feature relationships)
- **Scenarios Generated**: 100 across 4 severity levels
- **Training Time**: 10 minutes

### Financial Forecasting Models
| Target | Model | Test R² | RMSE |
|--------|-------|---------|------|
| Revenue | LightGBM Tuned | 0.9425 | 3.97B |
| EPS | LightGBM Tuned | 0.7036 | 1.37 |
| Debt-to-Equity | LightGBM Tuned | 0.6607 | 3.31 |
| Profit Margin | XGBoost Tuned | 0.4810 | 10.10 |
| Stock Returns | LightGBM Tuned | 0.0572 | 0.00 |
| **Average** | | **0.7847** | |

### Anomaly Detection (One-Class SVM)
- **ROC-AUC**: 0.8173 (96% of 0.85 target)
- **Precision@10%**: 0.67
- **Recall**: 0.783
- **At-Risk Rate**: 3.55% (realistic)
- **Sector Bias**: Zero (fair across all industries)

## Monitoring

The system includes comprehensive monitoring for:

- **Data Quality**: Great Expectations validation reports (94/100 score)
- **Model Performance**: MLflow tracking across all experiments
- **Data Drift**: Statistical distribution comparisons
- **Pipeline Health**: Airflow success rates and execution times
- **Bias Detection**: Three-tier framework with automated alerts

## Contributing

This is an academic project for an MLOps course. Team members follow standard Git workflow:

1. Create feature branch from `main`
2. Make changes and commit with descriptive messages
3. Run tests locally (`pytest tests/`)
4. Open pull request for review
5. Merge after approval and passing CI/CD

## Team

**MLOps Course Project - 6-person team**:
- Data Engineering Team: Pipeline development and data quality
- ML Engineering Team: Model development and training
- Infrastructure Team: Deployment and monitoring
- Project Lead: Coordination and documentation

## License

Academic project for MLOps course. All rights reserved.

## Acknowledgments

- **Data Sources**: Federal Reserve Economic Data (FRED), Yahoo Finance
- **ML Frameworks**: PyTorch, XGBoost, LightGBM, scikit-learn, Snorkel
- **MLOps Tools**: Apache Airflow, MLflow, DVC, Great Expectations, SHAP
- **Cloud Platform**: Google Cloud Platform
- **Course Instructors**: For guidance and feedback throughout the project
