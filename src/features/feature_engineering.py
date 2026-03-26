"""
Feature Engineering Pipeline
=============================
Transforms raw transaction data into model-ready features:
  - Scales numerical features (StandardScaler)
  - Creates interaction features capturing fraud signals
  - Encodes categorical variables
  - Splits into train/test sets with stratification
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SEED = 42
TEST_SIZE = 0.2

# Features that should be scaled
NUMERICAL_FEATURES = [
    "transaction_amount",
    "time_of_day",
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "velocity_last_hour",
    "velocity_last_24h",
    "avg_amount_last_7d",
    "hour_sin",
    "hour_cos",
]

# Binary / categorical features (no scaling needed)
BINARY_FEATURES = [
    "repeat_retailer",
    "used_chip",
    "used_pin",
    "online_order",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "day_of_week",
]

TARGET = "is_fraud"


def load_raw_data() -> pd.DataFrame:
    """Load raw transaction data from CSV."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}. "
            "Run `python -m src.data.generate_data` first."
        )
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} transactions with {df.columns.size} columns")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features that capture fraud signals."""
    # High amount + far from home = suspicious
    df["amount_x_distance"] = df["transaction_amount"] * df["distance_from_home"]

    # High velocity + high amount = card testing / cash-out
    df["velocity_x_amount"] = df["velocity_last_hour"] * df["transaction_amount"]

    # Online + no pin + no chip = higher risk
    df["no_security"] = (
        df["online_order"] * (1 - df["used_chip"]) * (1 - df["used_pin"])
    ).astype(int)

    # Amount deviation from recent average
    df["amount_deviation"] = (
        df["transaction_amount"] / df["avg_amount_last_7d"].clip(lower=1)
    )

    # Distance ratio: current vs last transaction
    df["distance_ratio"] = (
        df["distance_from_home"] / df["distance_from_last_transaction"].clip(lower=0.1)
    )

    # High velocity at unusual hours
    df["velocity_x_night"] = df["velocity_last_hour"] * (
        ((df["time_of_day"] < 6) | (df["time_of_day"] > 22)).astype(int)
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features with manageable cardinality."""
    for col in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)
    return df


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on training data and transform both sets."""
    # Identify which numerical columns are actually present after encoding
    num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    # Also scale the new interaction features
    interaction_cols = [
        "amount_x_distance",
        "velocity_x_amount",
        "amount_deviation",
        "distance_ratio",
        "velocity_x_night",
    ]
    cols_to_scale = num_cols + [c for c in interaction_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train, X_test, scaler


def run_pipeline() -> dict:
    """Execute the full feature engineering pipeline."""
    # 1. Load raw data
    df = load_raw_data()

    # 2. Create interaction features
    df = create_interaction_features(df)
    print(f"Created interaction features → {df.shape[1]} total columns")

    # 3. Encode categoricals
    df = encode_categoricals(df)
    print(f"Encoded categoricals → {df.shape[1]} total columns")

    # 4. Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 5. Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"Train set: {len(X_train):,} | Test set: {len(X_test):,}")
    print(f"Train fraud rate: {y_train.mean()*100:.2f}% | Test fraud rate: {y_test.mean()*100:.2f}%")

    # 6. Scale numerical features
    X_train, X_test, scaler = scale_features(X_train, X_test)
    print("Scaled numerical and interaction features")

    # 7. Save processed data and artifacts
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # Save feature list for API/inference
    feature_meta = {
        "feature_names": list(X_train.columns),
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(os.path.join(PROCESSED_DIR, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    print(f"\nProcessed data saved to: {PROCESSED_DIR}")
    print(f"Scaler saved to: {MODELS_DIR}/scaler.joblib")
    print(f"Feature count: {X_train.shape[1]}")

    return feature_meta


def main():
    """Run the feature engineering pipeline."""
    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)
    run_pipeline()


if __name__ == "__main__":
    main()
