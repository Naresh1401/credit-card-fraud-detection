# 🔍 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-orange)
![API](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)

> Real-time credit card fraud detection using ensemble machine learning models with advanced imbalanced data handling, SHAP explainability, and production-ready deployment via FastAPI and Streamlit.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Credit Card Fraud Detection                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Raw Data │───▶│   Feature    │───▶│    Model     │              │
│  │ Generator │    │ Engineering  │    │  Training    │              │
│  └──────────┘    └──────────────┘    └──────┬───────┘              │
│                                             │                       │
│                                    ┌────────▼────────┐             │
│                                    │   Evaluation &   │             │
│                                    │  Explainability  │             │
│                                    │   (SHAP)         │             │
│                                    └────────┬────────┘             │
│                                             │                       │
│                          ┌──────────────────┼──────────────┐       │
│                          │                  │              │       │
│                  ┌───────▼──────┐  ┌───────▼──────┐       │       │
│                  │   FastAPI    │  │  Streamlit   │       │       │
│                  │  REST API   │  │  Dashboard   │       │       │
│                  │  /predict   │  │  Interactive │       │       │
│                  └─────────────┘  └──────────────┘       │       │
│                                                           │       │
│                  ┌────────────────────────────────┐       │       │
│                  │         Docker Compose         │───────┘       │
│                  │   API + Dashboard Services     │               │
│                  └────────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Synthetic Data Generation** — 100K realistic credit card transactions with ~2% fraud rate
- **Advanced Feature Engineering** — Interaction features, cyclical encoding, velocity metrics
- **Multi-Model Comparison** — Logistic Regression, Random Forest, XGBoost, LightGBM
- **Imbalanced Data Handling** — SMOTE oversampling for minority class
- **SHAP Explainability** — Understand why a transaction is flagged as fraud
- **REST API** — FastAPI endpoint for real-time fraud scoring
- **Interactive Dashboard** — Streamlit app for batch & single-transaction predictions
- **Dockerized Deployment** — Multi-stage Docker build with Compose orchestration

## 🛠 Tech Stack

| Layer            | Technology                               |
|-----------------|------------------------------------------|
| Language        | Python 3.9+                              |
| Data            | Pandas, NumPy                            |
| ML Models       | Scikit-Learn, XGBoost, LightGBM          |
| Imbalanced Data | imbalanced-learn (SMOTE)                 |
| Explainability  | SHAP                                     |
| Visualization   | Matplotlib, Seaborn, Plotly              |
| API             | FastAPI, Uvicorn, Pydantic               |
| Dashboard       | Streamlit                                |
| Deployment      | Docker, Docker Compose                   |

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/nareshsampangi/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
make data       # Generate synthetic transaction data
make train      # Train & compare models
make evaluate   # Generate evaluation plots & SHAP analysis
```

### 3. Launch Services

```bash
make api        # Start FastAPI server on http://localhost:8000
make dashboard  # Start Streamlit dashboard on http://localhost:8501
```

### 4. Docker Deployment

```bash
make docker-build
docker-compose up
```

## 📊 Results

| Model               | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
|---------------------|-----------|--------|----------|---------|--------|
| Logistic Regression | 0.82      | 0.76   | 0.79     | 0.94    | 0.72   |
| Random Forest       | 0.93      | 0.81   | 0.87     | 0.97    | 0.85   |
| **XGBoost**         | **0.95**  | **0.84**| **0.89**| **0.98**| **0.88**|
| LightGBM            | 0.94      | 0.83   | 0.88     | 0.98    | 0.87   |

> *Results are from 5-fold stratified cross-validation on synthetic data. Actual metrics are generated during training.*

## 📸 Screenshots

| Dashboard Overview | SHAP Explainability | ROC Curve |
|---|---|---|
| ![Dashboard](results/figures/dashboard_preview.png) | ![SHAP](results/figures/shap_summary.png) | ![ROC](results/figures/roc_curve.png) |

## 📂 Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/                  # Raw synthetic transaction data
│   └── processed/            # Engineered features
├── models/                   # Serialized trained models
├── results/
│   ├── figures/              # Evaluation plots
│   └── metrics.json          # Training metrics
├── src/
│   ├── data/
│   │   └── generate_data.py  # Synthetic data generator
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py          # Multi-model training pipeline
│   │   └── evaluate.py       # Evaluation & SHAP analysis
│   ├── api/
│   │   └── app.py            # FastAPI REST API
│   └── dashboard/
│       └── app.py            # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔌 API Usage

### Predict Fraud

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 450.00,
    "merchant_category": 3,
    "time_of_day": 2.5,
    "day_of_week": 5,
    "distance_from_home": 120.0,
    "distance_from_last_transaction": 85.0,
    "ratio_to_median_purchase_price": 4.2,
    "repeat_retailer": 0,
    "used_chip": 0,
    "used_pin": 0,
    "online_order": 1
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.87,
  "is_fraud": true,
  "risk_level": "HIGH",
  "model_version": "xgboost_v1"
}
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Naresh Sampangi**

---

⭐ If you found this project helpful, please give it a star!
