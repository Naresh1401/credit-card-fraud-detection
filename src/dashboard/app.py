"""
Streamlit Fraud Detection Dashboard
=====================================
Interactive dashboard for:
  - Single-transaction fraud prediction
  - Batch CSV upload and scoring
  - Model performance metrics display
  - SHAP waterfall explanations
  - Transaction distribution charts
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    """Load the trained model (cached)."""
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data
def load_metrics():
    """Load training metrics (cached)."""
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return json.load(f)


@st.cache_data
def load_feature_meta():
    """Load feature metadata (cached)."""
    meta_path = os.path.join(PROCESSED_DIR, "feature_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def render_header():
    """Render the dashboard header."""
    st.title("🔍 Credit Card Fraud Detection")
    st.markdown(
        "Real-time fraud scoring powered by ensemble machine learning "
        "with SHAP explainability."
    )
    st.divider()


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Single Prediction", "Batch Predictions", "Model Performance", "Data Explorer"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.info("**Author:** Naresh Sampangi")
    return page


def render_single_prediction(model, feature_meta):
    """Render the single transaction prediction form."""
    st.header("🎯 Single Transaction Prediction")
    st.markdown("Enter transaction details to get a real-time fraud score.")

    col1, col2, col3 = st.columns(3)

    with col1:
        transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=150.0, step=10.0)
        merchant_category = st.selectbox("Merchant Category", list(range(15)), index=3)
        time_of_day = st.slider("Time of Day (Hour)", 0.0, 23.99, 14.0, 0.5)
        day_of_week = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

    with col2:
        distance_from_home = st.number_input("Distance from Home (km)", min_value=0.0, value=10.0, step=5.0)
        distance_from_last_txn = st.number_input("Distance from Last Transaction (km)", min_value=0.0, value=5.0, step=5.0)
        ratio_to_median = st.number_input("Ratio to Median Purchase Price", min_value=0.01, value=1.0, step=0.1)

    with col3:
        repeat_retailer = st.selectbox("Repeat Retailer?", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")
        used_chip = st.selectbox("Used Chip?", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")
        used_pin = st.selectbox("Used PIN?", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")
        online_order = st.selectbox("Online Order?", [0, 1], index=0, format_func=lambda x: "Yes" if x else "No")

    if st.button("🔎 Predict Fraud", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded. Please train the model first.")
            return

        # Build feature vector
        is_weekend = 1 if day_of_week >= 5 else 0
        hour_sin = np.sin(2 * np.pi * time_of_day / 24)
        hour_cos = np.cos(2 * np.pi * time_of_day / 24)

        velocity_last_hour = 1
        velocity_last_24h = 5
        avg_amount_last_7d = transaction_amount

        features = {
            "transaction_amount": transaction_amount,
            "time_of_day": time_of_day,
            "distance_from_home": distance_from_home,
            "distance_from_last_transaction": distance_from_last_txn,
            "ratio_to_median_purchase_price": ratio_to_median,
            "repeat_retailer": repeat_retailer,
            "used_chip": used_chip,
            "used_pin": used_pin,
            "online_order": online_order,
            "velocity_last_hour": velocity_last_hour,
            "velocity_last_24h": velocity_last_24h,
            "avg_amount_last_7d": avg_amount_last_7d,
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "amount_x_distance": transaction_amount * distance_from_home,
            "velocity_x_amount": velocity_last_hour * transaction_amount,
            "no_security": online_order * (1 - used_chip) * (1 - used_pin),
            "amount_deviation": transaction_amount / max(avg_amount_last_7d, 1),
            "distance_ratio": distance_from_home / max(distance_from_last_txn, 0.1),
            "velocity_x_night": velocity_last_hour * (1 if (time_of_day < 6 or time_of_day > 22) else 0),
        }

        # One-hot encode categoricals
        if feature_meta:
            for fname in feature_meta["feature_names"]:
                if fname.startswith("merchant_category_"):
                    cat_val = int(fname.split("_")[-1])
                    features[fname] = 1 if merchant_category == cat_val else 0
                elif fname.startswith("day_of_week_"):
                    day_val = int(fname.split("_")[-1])
                    features[fname] = 1 if day_of_week == day_val else 0

            ordered = [features.get(f, 0) for f in feature_meta["feature_names"]]
            X_input = np.array([ordered])
        else:
            X_input = np.array([list(features.values())])

        probability = float(model.predict_proba(X_input)[:, 1][0])
        is_fraud = probability >= 0.5

        # Display result
        st.divider()
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric("Fraud Probability", f"{probability:.1%}")
        with col_r2:
            if is_fraud:
                st.error(f"⚠️ FRAUD DETECTED")
            else:
                st.success(f"✅ Legitimate Transaction")
        with col_r3:
            if probability >= 0.7:
                st.metric("Risk Level", "🔴 HIGH")
            elif probability >= 0.3:
                st.metric("Risk Level", "🟡 MEDIUM")
            else:
                st.metric("Risk Level", "🟢 LOW")

        # SHAP explanation for this prediction
        try:
            clf = model.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
            else:
                explainer = shap.LinearExplainer(clf, X_input)

            shap_values = explainer.shap_values(X_input)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            if feature_meta:
                feature_names = feature_meta["feature_names"]
            else:
                feature_names = list(features.keys())

            st.subheader("📊 SHAP Explanation")
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                data=X_input[0],
                feature_names=feature_names,
            )
            fig, ax = shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig, clear_figure=True)
        except Exception:
            st.info("SHAP waterfall plot unavailable for this model type.")


def render_batch_predictions(model, feature_meta):
    """Render the batch prediction interface."""
    st.header("📁 Batch Predictions")
    st.markdown("Upload a CSV file with transaction data to score all rows.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(df):,}** transactions")
        st.dataframe(df.head(), use_container_width=True)

        if model is None:
            st.error("Model not loaded.")
            return

        if st.button("🚀 Run Batch Prediction", type="primary"):
            try:
                probas = model.predict_proba(df)[:, 1]
                df["fraud_probability"] = probas
                df["is_fraud"] = (probas >= 0.5).astype(int)
                df["risk_level"] = pd.cut(
                    probas,
                    bins=[0, 0.3, 0.7, 1.0],
                    labels=["LOW", "MEDIUM", "HIGH"],
                )

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", f"{len(df):,}")
                col2.metric("Flagged as Fraud", f"{df['is_fraud'].sum():,}")
                col3.metric("Fraud Rate", f"{df['is_fraud'].mean():.1%}")

                # Distribution chart
                fig = px.histogram(
                    df, x="fraud_probability", nbins=50,
                    title="Fraud Probability Distribution",
                    color_discrete_sequence=["#FF5722"],
                )
                st.plotly_chart(fig, use_container_width=True)

                # Results table
                st.subheader("Scored Transactions")
                st.dataframe(
                    df.sort_values("fraud_probability", ascending=False),
                    use_container_width=True,
                )

                # Download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv, "fraud_predictions.csv", "text/csv",
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


def render_model_performance(metrics):
    """Render model performance metrics and comparison."""
    st.header("📈 Model Performance")

    if metrics is None:
        st.warning("No metrics found. Train the model first.")
        return

    # Best model callout
    st.subheader(f"Best Model: {metrics['best_model']}")

    # Test set metrics
    test = metrics.get("test_set", {})
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{test.get('precision', 0):.4f}")
    col2.metric("Recall", f"{test.get('recall', 0):.4f}")
    col3.metric("F1-Score", f"{test.get('f1', 0):.4f}")
    col4.metric("AUC-ROC", f"{test.get('roc_auc', 0):.4f}")
    col5.metric("PR-AUC", f"{test.get('pr_auc', 0):.4f}")

    # Cross-validation comparison
    st.subheader("Cross-Validation Comparison")
    cv_data = metrics.get("cross_validation", {})

    if cv_data:
        rows = []
        for model_name, model_metrics in cv_data.items():
            rows.append({
                "Model": model_name,
                **{k.upper(): round(v, 4) for k, v in model_metrics.items()},
            })
        cv_df = pd.DataFrame(rows)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        melted = cv_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
        fig = px.bar(
            melted, x="Model", y="Score", color="Metric",
            barmode="group", title="Model Comparison",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # Show saved evaluation plots
    st.subheader("Evaluation Plots")
    plot_files = {
        "ROC Curve": "roc_curve.png",
        "Precision-Recall Curve": "precision_recall_curve.png",
        "Confusion Matrix": "confusion_matrix.png",
        "Feature Importance": "feature_importance.png",
        "SHAP Summary": "shap_summary.png",
    }

    cols = st.columns(2)
    for idx, (title, fname) in enumerate(plot_files.items()):
        fpath = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(fpath):
            with cols[idx % 2]:
                st.image(fpath, caption=title, use_container_width=True)


def render_data_explorer():
    """Render transaction data distribution charts."""
    st.header("🔬 Data Explorer")

    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
    if not os.path.exists(raw_path):
        st.warning("No data found. Generate data first.")
        return

    df = pd.read_csv(raw_path)

    # Overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraud Count", f"{df['is_fraud'].sum():,}")
    col3.metric("Fraud Rate", f"{df['is_fraud'].mean():.2%}")

    st.divider()

    # Amount distribution by class
    fig1 = px.histogram(
        df, x="transaction_amount", color="is_fraud",
        nbins=100, title="Transaction Amount Distribution",
        labels={"is_fraud": "Is Fraud"},
        color_discrete_map={0: "#4CAF50", 1: "#F44336"},
        marginal="box",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Time of day distribution
    fig2 = px.histogram(
        df, x="time_of_day", color="is_fraud",
        nbins=48, title="Transaction Time Distribution",
        labels={"is_fraud": "Is Fraud"},
        color_discrete_map={0: "#4CAF50", 1: "#F44336"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Distance from home
    col_a, col_b = st.columns(2)

    with col_a:
        fig3 = px.box(
            df, x="is_fraud", y="distance_from_home",
            title="Distance from Home by Class",
            labels={"is_fraud": "Is Fraud"},
            color="is_fraud",
            color_discrete_map={0: "#4CAF50", 1: "#F44336"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        fig4 = px.box(
            df, x="is_fraud", y="velocity_last_hour",
            title="Velocity (Last Hour) by Class",
            labels={"is_fraud": "Is Fraud"},
            color="is_fraud",
            color_discrete_map={0: "#4CAF50", 1: "#F44336"},
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    fig5 = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r",
    )
    st.plotly_chart(fig5, use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    render_header()
    page = render_sidebar()

    model = load_model()
    metrics = load_metrics()
    feature_meta = load_feature_meta()

    if page == "Single Prediction":
        render_single_prediction(model, feature_meta)
    elif page == "Batch Predictions":
        render_batch_predictions(model, feature_meta)
    elif page == "Model Performance":
        render_model_performance(metrics)
    elif page == "Data Explorer":
        render_data_explorer()


if __name__ == "__main__":
    main()
