"""
Classify additional data embeddings using existing corpus-trained Ridge models.

Applies the same quality scoring used for the main corpus to additional data
sources (NYT, Economist, FT, Newswire). Uses pre-trained Ridge models from
D:/hist_LLM/processing/quality_models/.

The Ridge models were trained on corpus data but the quality signal (coherence,
depth, writing quality) transfers via BGE embeddings in the same vector space.

Usage:
    python classify_additional.py                    # Classify all collections
    python classify_additional.py --collection nyt   # Classify specific collection
    python classify_additional.py --status           # Show status only

Output:
    D:/hist_LLM/additional_data/classified/{collection}/classified_{year}.parquet
    Columns: [identifier, predicted_quality]
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
EMBEDDINGS_DIR = Path(r"D:\hist_LLM\additional_data\embeddings")
MODELS_DIR = Path(r"D:\hist_LLM\processing\quality_models")
OUTPUT_DIR = Path(r"D:\hist_LLM\additional_data\classified")

# Collections with their embedding directory names
COLLECTIONS = ["nyt", "economist", "ft", "newswire"]

# 25-year periods (same as check_and_classify.py)
PERIODS = [
    (1678, 1700, "1678_1700"),
    (1701, 1725, "1701_1725"),
    (1726, 1750, "1726_1750"),
    (1751, 1775, "1751_1775"),
    (1776, 1800, "1776_1800"),
    (1801, 1825, "1801_1825"),
    (1826, 1850, "1826_1850"),
    (1851, 1875, "1851_1875"),
    (1876, 1900, "1876_1900"),
    (1901, 1925, "1901_1925"),
    (1926, 1950, "1926_1950"),
    (1951, 1975, "1951_1975"),
    (1976, 2000, "1976_2000"),
    (2001, 2023, "2001_2023"),
]


def get_period_for_year(year):
    """Map a year to its 25-year quality model period."""
    for start, end, period_name in PERIODS:
        if start <= year <= end:
            return period_name
    return None


def load_models(period):
    """Load scaler and Ridge model for a period."""
    scaler_path = MODELS_DIR / f"scaler_{period}.pkl"
    model_path = MODELS_DIR / f"ridge_{period}.pkl"
    if not scaler_path.exists() or not model_path.exists():
        return None
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model


def get_status():
    """Get classification status for all collections."""
    status = {}
    for collection in COLLECTIONS:
        emb_dir = EMBEDDINGS_DIR / collection
        out_dir = OUTPUT_DIR / collection

        if not emb_dir.exists():
            status[collection] = {"years": [], "classified": [], "pending": []}
            continue

        years = sorted(
            int(f.stem.replace("embeddings_", ""))
            for f in emb_dir.glob("embeddings_*.parquet")
        )
        classified = []
        if out_dir.exists():
            classified = sorted(
                int(f.stem.replace("classified_", ""))
                for f in out_dir.glob("classified_*.parquet")
            )

        pending = [y for y in years if y not in classified]
        status[collection] = {
            "years": years,
            "classified": classified,
            "pending": pending,
        }

    return status


def print_status(status):
    """Print classification status."""
    print(f"\n{'Collection':<15} {'Embeddings':>10} {'Classified':>12} {'Pending':>10}")
    print("-" * 50)
    for collection, info in sorted(status.items()):
        print(f"{collection:<15} {len(info['years']):>10} "
              f"{len(info['classified']):>12} {len(info['pending']):>10}")
    print()


def classify_collection(collection, reclassify=False):
    """Classify all years for a collection."""
    emb_dir = EMBEDDINGS_DIR / collection
    out_dir = OUTPUT_DIR / collection
    out_dir.mkdir(parents=True, exist_ok=True)

    if not emb_dir.exists():
        print(f"  {collection}: no embeddings directory found, skipping")
        return

    years = sorted(
        int(f.stem.replace("embeddings_", ""))
        for f in emb_dir.glob("embeddings_*.parquet")
    )

    if not reclassify:
        existing = set(
            int(f.stem.replace("classified_", ""))
            for f in out_dir.glob("classified_*.parquet")
        )
        years = [y for y in years if y not in existing]

    if not years:
        print(f"  {collection}: all years already classified")
        return

    print(f"  {collection}: classifying {len(years)} years...")

    # Cache models by period to avoid reloading
    model_cache = {}
    total_docs = 0
    total_years = 0

    for year in years:
        period = get_period_for_year(year)
        if period is None:
            print(f"    year {year}: no quality model period, skipping")
            continue

        # Load model (cached)
        if period not in model_cache:
            models = load_models(period)
            if models is None:
                print(f"    year {year}: no model for period {period}, skipping")
                model_cache[period] = None
                continue
            model_cache[period] = models

        if model_cache[period] is None:
            continue

        scaler, model = model_cache[period]

        # Load embeddings
        emb_path = emb_dir / f"embeddings_{year}.parquet"
        emb_df = pd.read_parquet(emb_path)

        if len(emb_df) == 0:
            continue

        if "embedding" not in emb_df.columns:
            print(f"    year {year}: no 'embedding' column, skipping")
            continue

        # Predict quality scores
        X = np.stack(emb_df["embedding"].values)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        # Use original_index as identifier
        identifiers = emb_df["original_index"].astype(str).tolist()

        result_df = pd.DataFrame({
            "identifier": identifiers,
            "predicted_quality": predictions,
        })

        # Save
        out_path = out_dir / f"classified_{year}.parquet"
        result_df.to_parquet(out_path, index=False)

        total_docs += len(result_df)
        total_years += 1

    print(f"    Done: {total_years} years, {total_docs:,} documents classified")


def main():
    parser = argparse.ArgumentParser(
        description="Classify additional data embeddings using corpus-trained Ridge models"
    )
    parser.add_argument("--collection", type=str, nargs="+", default=None,
                        choices=COLLECTIONS,
                        help="Specific collections to classify (default: all)")
    parser.add_argument("--status", action="store_true",
                        help="Show classification status only")
    parser.add_argument("--reclassify", action="store_true",
                        help="Reclassify all years (overwrite existing)")
    args = parser.parse_args()

    status = get_status()
    print_status(status)

    if args.status:
        return

    collections = args.collection or COLLECTIONS

    print(f"Classifying additional data using corpus-trained Ridge models")
    print(f"Models: {MODELS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for collection in collections:
        classify_collection(collection, reclassify=args.reclassify)

    print("\nDone.")


if __name__ == "__main__":
    main()
