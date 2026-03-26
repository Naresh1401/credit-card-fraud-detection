"""
Model Training Pipeline
========================
Trains and compares multiple classifiers for fraud detection:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
  - LightGBM

Uses SMOTE to handle class imbalance and stratified k-fold
cross-validation for robust evaluation.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from tabulate import tabulate
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    make_scorer,
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

SEED = 42
N_FOLDS = 5


def load_processed_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the processed train/test splits."""
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    return X_train, y_train, X_test, y_test


def get_models() -> dict:
    """Define the model candidates with tuned hyperparameters."""
    return {
        "Logistic Regression": ImbPipeline([
            ("smote", SMOTE(random_state=SEED)),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=0.1,
                class_weight="balanced",
                random_state=SEED,
            )),
        ]),
        "Random Forest": ImbPipeline([
            ("smote", SMOTE(random_state=SEED)),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": ImbPipeline([
            ("smote", SMOTE(random_state=SEED)),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # SMOTE handles imbalance
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
        "LightGBM": ImbPipeline([
            ("smote", SMOTE(random_state=SEED)),
            ("clf", LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
    }


def get_scoring() -> dict:
    """Define the scoring metrics for cross-validation."""
    return {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(average_precision_score, needs_proba=True),
    }


def train_and_evaluate(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[dict, dict]:
    """Train all models with cross-validation and return results."""
    models = get_models()
    scoring = get_scoring()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    results = {}
    trained_models = {}

    print(f"\nRunning {N_FOLDS}-fold stratified cross-validation...\n")

    for name, pipeline in models.items():
        print(f"  Training {name}...", end=" ", flush=True)

        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        results[name] = {
            "precision": float(np.mean(cv_results["test_precision"])),
            "recall": float(np.mean(cv_results["test_recall"])),
            "f1": float(np.mean(cv_results["test_f1"])),
            "roc_auc": float(np.mean(cv_results["test_roc_auc"])),
            "pr_auc": float(np.mean(cv_results["test_pr_auc"])),
        }

        # Fit on full training set for final model
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

        print(f"F1={results[name]['f1']:.4f}  AUC-ROC={results[name]['roc_auc']:.4f}")

    return results, trained_models


def select_best_model(
    results: dict, trained_models: dict
) -> tuple[str, object]:
    """Select the best model based on F1 score."""
    best_name = max(results, key=lambda k: results[k]["f1"])
    return best_name, trained_models[best_name]


def evaluate_on_test(
    model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
) -> dict:
    """Evaluate the best model on the held-out test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }

    print(f"\n{'='*60}")
    print(f"Best Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Test Precision : {test_metrics['precision']:.4f}")
    print(f"  Test Recall    : {test_metrics['recall']:.4f}")
    print(f"  Test F1        : {test_metrics['f1']:.4f}")
    print(f"  Test AUC-ROC   : {test_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC    : {test_metrics['pr_auc']:.4f}")

    return test_metrics


def print_comparison_table(results: dict):
    """Print a formatted comparison table of all models."""
    rows = []
    for name, metrics in results.items():
        rows.append([
            name,
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['pr_auc']:.4f}",
        ])

    headers = ["Model", "Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC"]
    print(f"\n{'='*60}")
    print("Model Comparison (Cross-Validation Averages)")
    print("=" * 60)
    try:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    except ImportError:
        # Fallback if tabulate not installed
        print(f"{'Model':<25} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC-ROC':>8} {'PR-AUC':>8}")
        print("-" * 75)
        for row in rows:
            print(f"{row[0]:<25} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8} {row[5]:>8}")


def save_artifacts(
    best_model,
    best_name: str,
    cv_results: dict,
    test_metrics: dict,
    feature_names: list,
):
    """Save the best model and all metrics to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save best model
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)

    # Save all metrics
    metrics_output = {
        "best_model": best_name,
        "cross_validation": cv_results,
        "test_set": test_metrics,
        "feature_names": feature_names,
        "n_folds": N_FOLDS,
        "seed": SEED,
    }
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\nArtifacts saved:")
    print(f"  Model   → {model_path}")
    print(f"  Metrics → {metrics_path}")


def main():
    """Run the full training pipeline."""
    print("=" * 60)
    print("Model Training Pipeline")
    print("=" * 60)

    # Load data
    X_train, y_train, X_test, y_test = load_processed_data()
    print(f"Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Test set:     {X_test.shape[0]:,} samples")
    print(f"Fraud rate:   {y_train.mean()*100:.2f}%")

    # Train and cross-validate all models
    cv_results, trained_models = train_and_evaluate(X_train, y_train)
    print_comparison_table(cv_results)

    # Select and evaluate best model on test set
    best_name, best_model = select_best_model(cv_results, trained_models)
    test_metrics = evaluate_on_test(best_model, X_test, y_test, best_name)

    # Save everything
    save_artifacts(
        best_model,
        best_name,
        cv_results,
        test_metrics,
        feature_names=list(X_train.columns),
    )


if __name__ == "__main__":
    main()
