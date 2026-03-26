"""
FastAPI Fraud Detection Service
================================
Production-ready REST API for real-time credit card fraud scoring.

Endpoints:
  POST /predict    — Score a single transaction
  GET  /health     — Service health check
  GET  /model/info — Model metadata and feature list
"""

import os
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ── Globals populated at startup ──────────────────────────────────────
model = None
scaler = None
feature_meta = None
metrics = None


def load_model_artifacts():
    """Load the trained model, scaler, and metadata."""
    global model, scaler, feature_meta, metrics

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    meta_path = os.path.join(PROCESSED_DIR, "feature_meta.json")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            feature_meta = json.load(f)

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    load_model_artifacts()
    yield


# ── FastAPI App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud scoring for credit card transactions",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic Models ───────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """Input schema for a credit card transaction."""
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: int = Field(..., ge=0, le=14, description="Merchant category code (0-14)")
    time_of_day: float = Field(..., ge=0, lt=24, description="Hour of transaction (0-23.99)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    distance_from_home: float = Field(..., ge=0, description="Distance from cardholder home (km)")
    distance_from_last_transaction: float = Field(..., ge=0, description="Distance from last transaction (km)")
    ratio_to_median_purchase_price: float = Field(..., gt=0, description="Ratio to cardholder median purchase")
    repeat_retailer: int = Field(..., ge=0, le=1, description="1 if repeat retailer, 0 otherwise")
    used_chip: int = Field(..., ge=0, le=1, description="1 if chip was used, 0 otherwise")
    used_pin: int = Field(..., ge=0, le=1, description="1 if PIN was used, 0 otherwise")
    online_order: int = Field(..., ge=0, le=1, description="1 if online order, 0 otherwise")

    model_config = {"json_schema_extra": {
        "examples": [{
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
            "online_order": 1,
        }]
    }}


class PredictionResponse(BaseModel):
    """Output schema for fraud prediction."""
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="True if predicted as fraud")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    model_version: str = Field(..., description="Model identifier")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str
    n_features: int
    feature_names: list[str]
    test_metrics: dict | None
    training_seed: int


# ── Helper Functions ──────────────────────────────────────────────────
def build_feature_vector(txn: TransactionRequest) -> np.ndarray:
    """
    Transform a transaction request into the feature vector expected by the model.
    Mirrors the feature engineering pipeline steps.
    """
    # Base features
    is_weekend = 1 if txn.day_of_week >= 5 else 0
    hour_sin = np.sin(2 * np.pi * txn.time_of_day / 24)
    hour_cos = np.cos(2 * np.pi * txn.time_of_day / 24)

    # Interaction features (using sensible defaults for velocity/avg fields)
    velocity_last_hour = 1
    velocity_last_24h = 5
    avg_amount_last_7d = txn.transaction_amount  # Default: assume typical

    amount_x_distance = txn.transaction_amount * txn.distance_from_home
    velocity_x_amount = velocity_last_hour * txn.transaction_amount
    no_security = txn.online_order * (1 - txn.used_chip) * (1 - txn.used_pin)
    amount_deviation = txn.transaction_amount / max(avg_amount_last_7d, 1)
    distance_ratio = txn.distance_from_home / max(txn.distance_from_last_transaction, 0.1)
    is_night = 1 if (txn.time_of_day < 6 or txn.time_of_day > 22) else 0
    velocity_x_night = velocity_last_hour * is_night

    # Build feature dict matching the training feature order
    features = {
        "transaction_amount": txn.transaction_amount,
        "time_of_day": txn.time_of_day,
        "distance_from_home": txn.distance_from_home,
        "distance_from_last_transaction": txn.distance_from_last_transaction,
        "ratio_to_median_purchase_price": txn.ratio_to_median_purchase_price,
        "repeat_retailer": txn.repeat_retailer,
        "used_chip": txn.used_chip,
        "used_pin": txn.used_pin,
        "online_order": txn.online_order,
        "velocity_last_hour": velocity_last_hour,
        "velocity_last_24h": velocity_last_24h,
        "avg_amount_last_7d": avg_amount_last_7d,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "amount_x_distance": amount_x_distance,
        "velocity_x_amount": velocity_x_amount,
        "no_security": no_security,
        "amount_deviation": amount_deviation,
        "distance_ratio": distance_ratio,
        "velocity_x_night": velocity_x_night,
    }

    # One-hot encode merchant_category and day_of_week to match training
    if feature_meta:
        for fname in feature_meta["feature_names"]:
            if fname.startswith("merchant_category_"):
                cat_val = int(fname.split("_")[-1])
                features[fname] = 1 if txn.merchant_category == cat_val else 0
            elif fname.startswith("day_of_week_"):
                day_val = int(fname.split("_")[-1])
                features[fname] = 1 if txn.day_of_week == day_val else 0

        # Reorder to match training feature order
        ordered = [features.get(f, 0) for f in feature_meta["feature_names"]]
        return np.array([ordered])

    # Fallback: return core features as array
    return np.array([list(features.values())])


def get_risk_level(probability: float) -> str:
    """Classify fraud probability into risk levels."""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    return "LOW"


# ── Endpoints ─────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    """Score a transaction for fraud probability."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = build_feature_vector(transaction)
    probability = float(model.predict_proba(features)[:, 1][0])
    is_fraud = probability >= 0.5

    model_name = metrics["best_model"] if metrics else "unknown"

    return PredictionResponse(
        fraud_probability=round(probability, 4),
        is_fraud=is_fraud,
        risk_level=get_risk_level(probability),
        model_version=f"{model_name.lower().replace(' ', '_')}_v1",
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Service health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata and performance metrics."""
    if model is None or feature_meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_name = metrics["best_model"] if metrics else "unknown"

    return ModelInfoResponse(
        model_name=model_name,
        n_features=feature_meta["n_features"],
        feature_names=feature_meta["feature_names"],
        test_metrics=metrics.get("test_set") if metrics else None,
        training_seed=42,
    )
