"""
Synthetic Credit Card Transaction Data Generator
=================================================
Generates 100,000 realistic credit card transactions with ~2% fraud rate.
Uses domain knowledge to create plausible fraud patterns including unusual
amounts, atypical times, and suspicious velocity metrics.
"""

import os
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────
SEED = 42
N_TRANSACTIONS = 100_000
FRAUD_RATE = 0.02
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "transactions.csv")


def generate_legitimate_transactions(n: int, rng: np.random.Generator) -> dict:
    """Generate features for legitimate (non-fraud) transactions."""
    return {
        "transaction_amount": rng.lognormal(mean=3.5, sigma=1.0, size=n).clip(0.50, 5000),
        "merchant_category": rng.integers(0, 15, size=n),
        "time_of_day": rng.normal(loc=14.0, scale=4.0, size=n).clip(0, 23.99),
        "day_of_week": rng.integers(0, 7, size=n),
        "distance_from_home": rng.exponential(scale=10.0, size=n).clip(0, 200),
        "distance_from_last_transaction": rng.exponential(scale=5.0, size=n).clip(0, 150),
        "ratio_to_median_purchase_price": rng.lognormal(mean=0.0, sigma=0.4, size=n).clip(0.1, 10),
        "repeat_retailer": rng.choice([0, 1], size=n, p=[0.2, 0.8]),
        "used_chip": rng.choice([0, 1], size=n, p=[0.3, 0.7]),
        "used_pin": rng.choice([0, 1], size=n, p=[0.4, 0.6]),
        "online_order": rng.choice([0, 1], size=n, p=[0.65, 0.35]),
        "velocity_last_hour": rng.poisson(lam=1.2, size=n),
        "velocity_last_24h": rng.poisson(lam=5.0, size=n),
        "avg_amount_last_7d": rng.lognormal(mean=3.5, sigma=0.8, size=n).clip(5, 3000),
    }


def generate_fraudulent_transactions(n: int, rng: np.random.Generator) -> dict:
    """Generate features for fraudulent transactions with realistic fraud patterns."""
    return {
        # Fraud tends to have higher amounts
        "transaction_amount": rng.lognormal(mean=4.5, sigma=1.5, size=n).clip(1, 15000),
        "merchant_category": rng.integers(0, 15, size=n),
        # Fraud skews toward late-night / early-morning hours
        "time_of_day": rng.normal(loc=3.0, scale=6.0, size=n) % 24,
        "day_of_week": rng.integers(0, 7, size=n),
        # Fraud happens farther from home and farther from last transaction
        "distance_from_home": rng.exponential(scale=60.0, size=n).clip(0, 500),
        "distance_from_last_transaction": rng.exponential(scale=40.0, size=n).clip(0, 500),
        # Unusually high ratio to median purchase price
        "ratio_to_median_purchase_price": rng.lognormal(mean=1.0, sigma=0.8, size=n).clip(0.5, 30),
        # Fraudsters less likely at repeat retailers, less likely to use chip/pin
        "repeat_retailer": rng.choice([0, 1], size=n, p=[0.75, 0.25]),
        "used_chip": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        "used_pin": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
        "online_order": rng.choice([0, 1], size=n, p=[0.3, 0.7]),
        # Higher velocity — rapid succession of transactions
        "velocity_last_hour": rng.poisson(lam=4.5, size=n),
        "velocity_last_24h": rng.poisson(lam=12.0, size=n),
        "avg_amount_last_7d": rng.lognormal(mean=3.5, sigma=0.8, size=n).clip(5, 3000),
    }


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical and calendar-derived features."""
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24)
    return df


def generate_dataset() -> pd.DataFrame:
    """Generate the full synthetic dataset with fraud labels."""
    rng = np.random.default_rng(SEED)

    n_fraud = int(N_TRANSACTIONS * FRAUD_RATE)
    n_legit = N_TRANSACTIONS - n_fraud

    print(f"Generating {n_legit:,} legitimate and {n_fraud:,} fraudulent transactions...")

    legit_data = generate_legitimate_transactions(n_legit, rng)
    fraud_data = generate_fraudulent_transactions(n_fraud, rng)

    df_legit = pd.DataFrame(legit_data)
    df_legit["is_fraud"] = 0

    df_fraud = pd.DataFrame(fraud_data)
    df_fraud["is_fraud"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = add_derived_features(df)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Round floats to reasonable precision
    float_cols = df.select_dtypes(include=[np.floating]).columns
    df[float_cols] = df[float_cols].round(4)

    return df


def main():
    """Generate and save synthetic transaction data."""
    df = generate_dataset()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")
    print(f"  Total transactions : {len(df):,}")
    print(f"  Fraud transactions : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"  Legit transactions : {(~df['is_fraud'].astype(bool)).sum():,}")
    print(f"  Features           : {len(df.columns) - 1}")
    print(f"  Columns            : {list(df.columns)}")
    print(f"\nSample stats:")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
