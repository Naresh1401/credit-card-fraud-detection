"""
Model Evaluation & Explainability
===================================
Generates comprehensive evaluation artifacts:
  - Confusion matrix heatmap
  - ROC curve
  - Precision-Recall curve
  - Feature importance (model-native + SHAP)
  - SHAP summary and dependence plots
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    classification_report,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Plot styling
plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE = (10, 7)
DPI = 150


def load_artifacts():
    """Load the best model, test data, and metadata."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    with open(os.path.join(RESULTS_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    return model, X_test, y_test, metrics


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)

    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_roc_curve(y_true, y_proba, save_path: str):
    """Generate and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_precision_recall_curve(y_true, y_proba, save_path: str):
    """Generate and save the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(recall, precision, color="#FF5722", lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#FF5722")

    # Baseline: fraction of positives
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", lw=1, linestyle="--", label=f"Baseline ({baseline:.4f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_feature_importance(model, feature_names: list, save_path: str):
    """Extract and plot feature importance from the underlying classifier."""
    # Get the classifier from the imblearn pipeline
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        print("  Skipping feature importance — model has no importances attribute")
        return

    # Sort by importance
    indices = np.argsort(importances)[::-1][:20]  # Top 20
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_importances[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Top 20 Feature Importances", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_shap_summary(model, X_test: pd.DataFrame, save_path: str):
    """Generate SHAP summary plot for global feature importance."""
    clf = model.named_steps["clf"]

    # Use TreeExplainer for tree-based models, otherwise KernelExplainer
    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
        # Use a sample for efficiency
        X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
    else:
        X_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
        explainer = shap.LinearExplainer(clf, X_sample)
        shap_values = explainer.shap_values(X_sample)

    # For binary classification, some explainers return a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (fraud)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title("SHAP Feature Impact on Fraud Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=DPI)
    plt.close("all")
    print(f"  Saved: {save_path}")


def plot_shap_bar(model, X_test: pd.DataFrame, save_path: str):
    """Generate SHAP bar plot for mean absolute SHAP values."""
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
        X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
    else:
        X_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
        explainer = shap.LinearExplainer(clf, X_sample)
        shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP Value| (Feature Importance)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=DPI)
    plt.close("all")
    print(f"  Saved: {save_path}")


def main():
    """Run the full evaluation pipeline."""
    print("=" * 60)
    print("Model Evaluation & Explainability")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load artifacts
    model, X_test, y_test, metrics = load_artifacts()
    best_name = metrics["best_model"]
    print(f"Evaluating: {best_name}\n")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Generate all plots
    print("Generating evaluation plots...")

    plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(FIGURES_DIR, "confusion_matrix.png"),
    )

    plot_roc_curve(
        y_test, y_proba,
        os.path.join(FIGURES_DIR, "roc_curve.png"),
    )

    plot_precision_recall_curve(
        y_test, y_proba,
        os.path.join(FIGURES_DIR, "precision_recall_curve.png"),
    )

    plot_feature_importance(
        model, list(X_test.columns),
        os.path.join(FIGURES_DIR, "feature_importance.png"),
    )

    print("\nGenerating SHAP explainability plots...")

    plot_shap_summary(
        model, X_test,
        os.path.join(FIGURES_DIR, "shap_summary.png"),
    )

    plot_shap_bar(
        model, X_test,
        os.path.join(FIGURES_DIR, "shap_bar.png"),
    )

    print(f"\nAll evaluation artifacts saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
