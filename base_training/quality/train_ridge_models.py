"""
Train Ridge models for each 25-year period.

This script:
1. Loads labeled embeddings for each 25-year period
2. Trains StandardScaler + RidgeCV pipeline for each period
3. Saves both scaler and model as .pkl files for later inference

Periods (14 total, 25-year windows):
    1678-1700, 1701-1725, 1726-1750, 1751-1775, 1776-1800, 1801-1825, 1826-1850,
    1851-1875, 1876-1900, 1901-1925, 1926-1950, 1951-1975, 1976-2000, 2001-2023

Output:
    Data/Classify_Data/Models/
    ├── ridge_1678_1700.pkl
    ├── scaler_1678_1700.pkl
    ...
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# --- CONFIG ---
EMBEDDING_DIR = Path(r"D:\hist_LLM\processing\labeled_embeddings")
MODEL_DIR = Path(r"D:\hist_LLM\processing\quality_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Original 25-year periods
PERIODS = [
    "1678_1700", "1701_1725", "1726_1750", "1751_1775",
    "1776_1800", "1801_1825", "1826_1850", "1851_1875",
    "1876_1900", "1901_1925", "1926_1950", "1951_1975",
    "1976_2000", "2001_2023",
]


def train_and_save_models():
    """Train Ridge models for each 25-year period and save them."""

    results_summary = []

    for period in PERIODS:
        print(f"\n{'='*60}")
        print(f"Training Ridge for period: {period}")
        print(f"{'='*60}")

        # Load labeled embeddings
        embedding_file = EMBEDDING_DIR / f"embeddings_bge_{period}.parquet"

        if not embedding_file.exists():
            print(f"  [SKIP] File not found: {embedding_file.name}")
            continue

        df = pd.read_parquet(embedding_file)
        print(f"  Loaded {len(df)} samples")

        X = np.stack(df['embedding'].values)
        y = df['labels'].values

        print(f"  Data shape: {X.shape}")
        print(f"  Label distribution: {np.bincount(y.astype(int), minlength=6)[1:]}")  # Counts for 1-5

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit StandardScaler on training data only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train RidgeCV (automatically finds best alpha)
        ridge = RidgeCV(
            alphas=np.logspace(-6, 6, 50),
            cv=10,
            scoring='r2'
        )
        ridge.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = ridge.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        corr, _ = pearsonr(y_test, y_pred)

        print(f"  Best alpha: {ridge.alpha_:.6f}")
        print(f"  Test MSE: {mse:.4f}")
        print(f"  Test R2: {r2:.4f}")
        print(f"  Test Pearson: {corr:.4f}")

        # Save scaler and model
        scaler_path = MODEL_DIR / f"scaler_{period}.pkl"
        model_path = MODEL_DIR / f"ridge_{period}.pkl"

        joblib.dump(scaler, scaler_path)
        joblib.dump(ridge, model_path)

        print(f"  Saved: {scaler_path.name}")
        print(f"  Saved: {model_path.name}")

        # Record results
        results_summary.append({
            "period": period,
            "n_samples": len(df),
            "best_alpha": ridge.alpha_,
            "test_mse": mse,
            "test_r2": r2,
            "test_pearson": corr
        })

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = MODEL_DIR / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Summary saved to: {summary_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    train_and_save_models()
