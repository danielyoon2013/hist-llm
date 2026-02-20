"""
Check embedding/classification status and classify any new embeddings.

Run this script periodically to:
1. See current status (embeddings done, classified, missing)
2. Automatically classify any newly completed embeddings

Usage:
    python check_and_classify.py          # Check status and classify new
    python check_and_classify.py --status # Only show status, don't classify
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
RAW_DIR = Path(r"D:\English")
EMBEDDINGS_DIR = Path(r"D:\English_Results")
MASKS_DIR = Path(r"D:\hist_LLM\Clean_Data\cleaning_masks")
MODELS_DIR = Path(r"D:\hist_LLM\Classify_Data\Models")
OUTPUT_DIR = Path(r"D:\English_Classified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Original 25-year periods
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


def get_period_for_year(year: int) -> str:
    """Map a year to its 25-year period model."""
    for start, end, period_name in PERIODS:
        if start <= year <= end:
            return period_name
    return None


def get_status():
    """Get current status of all years."""

    # All years from raw data
    all_years = sorted([int(d.name) for d in RAW_DIR.iterdir()
                        if d.is_dir() and d.name.isdigit()])

    # Years with embeddings
    emb_years = set()
    for f in EMBEDDINGS_DIR.glob('embeddings_*.parquet'):
        try:
            year = int(f.stem.split('_')[1])
            emb_years.add(year)
        except:
            pass

    # Years already classified
    classified_years = set()
    for f in OUTPUT_DIR.glob('classified_*.parquet'):
        try:
            year = int(f.stem.split('_')[1])
            classified_years.add(year)
        except:
            pass

    # Categorize
    status = {
        'all_years': set(all_years),
        'has_embedding': emb_years,
        'classified': classified_years,
        'ready_to_classify': emb_years - classified_years,  # Has embedding but not classified
        'missing_embedding': set(all_years) - emb_years,    # No embedding yet
    }

    return status


def print_status(status):
    """Print status summary."""

    print("=" * 60)
    print(f"STATUS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    total = len(status['all_years'])
    has_emb = len(status['has_embedding'])
    classified = len(status['classified'])
    ready = len(status['ready_to_classify'])
    missing = len(status['missing_embedding'])

    print(f"\nTotal years: {total}")
    print(f"  Embeddings done: {has_emb} ({100*has_emb/total:.1f}%)")
    print(f"    - Classified: {classified}")
    print(f"    - Ready to classify: {ready}")
    print(f"  Missing embeddings: {missing} ({100*missing/total:.1f}%)")

    if status['ready_to_classify']:
        ready_list = sorted(status['ready_to_classify'])
        print(f"\nReady to classify: {ready_list[:10]}{'...' if len(ready_list) > 10 else ''}")

    if status['missing_embedding']:
        missing_list = sorted(status['missing_embedding'])
        print(f"\nMissing embeddings (first 15): {missing_list[:15]}")
        print(f"Missing embeddings (last 15): {missing_list[-15:]}")

    print()


def load_models(period: str):
    """Load scaler and ridge model for a period."""
    scaler = joblib.load(MODELS_DIR / f"scaler_{period}.pkl")
    model = joblib.load(MODELS_DIR / f"ridge_{period}.pkl")
    return scaler, model


def load_clean_indices_for_year(year: int) -> dict:
    """Load all cleaning masks for a year."""
    year_mask_dir = MASKS_DIR / str(year)
    if not year_mask_dir.exists():
        return {}

    clean_indices = {}
    for mask_file in year_mask_dir.glob("*_mask.parquet"):
        subset_name = mask_file.name.replace("_mask.parquet", ".parquet")
        mask_df = pd.read_parquet(mask_file)
        clean_indices[subset_name] = set(mask_df['original_index'].tolist())

    return clean_indices


def classify_year(year: int, scaler, model) -> pd.DataFrame:
    """Classify all clean documents for a single year."""

    # Load embeddings
    emb_path = EMBEDDINGS_DIR / f"embeddings_{year}.parquet"
    emb_df = pd.read_parquet(emb_path)

    # Load cleaning masks
    clean_indices = load_clean_indices_for_year(year)
    if not clean_indices:
        return None

    # Filter to clean rows
    def is_clean_row(row):
        subset = row['subset_source']
        idx = row['row_idx']
        return subset in clean_indices and idx in clean_indices[subset]

    mask = emb_df.apply(is_clean_row, axis=1)
    clean_df = emb_df[mask].copy()

    if len(clean_df) == 0:
        return None

    # Predict
    X = np.stack(clean_df['embedding'].values)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Get token_count and word_count from raw data
    raw_dir = RAW_DIR / str(year)
    token_counts = {}
    word_counts = {}

    for subset_name in clean_df['subset_source'].unique():
        subset_path = raw_dir / subset_name
        if subset_path.exists():
            raw_df = pd.read_parquet(subset_path, columns=['identifier', 'token_count', 'word_count'])
            for _, row in raw_df.iterrows():
                token_counts[row['identifier']] = row['token_count']
                word_counts[row['identifier']] = row['word_count']

    # Build result
    result_df = pd.DataFrame({
        'identifier': clean_df['original_index'].values,
        'predicted_quality': predictions,
        'is_clean': True,
        'token_count': [token_counts.get(i, None) for i in clean_df['original_index'].values],
        'word_count': [word_counts.get(i, None) for i in clean_df['original_index'].values]
    })

    return result_df


def classify_new(status):
    """Classify any years that are ready."""

    ready_years = sorted(status['ready_to_classify'])

    if not ready_years:
        print("No new years to classify.")
        return

    print(f"Classifying {len(ready_years)} years...")

    model_cache = {}

    for year in ready_years:
        period = get_period_for_year(year)
        if not period:
            print(f"  [SKIP] {year}: No period mapping")
            continue

        # Load model (cached)
        if period not in model_cache:
            try:
                model_cache[period] = load_models(period)
            except Exception as e:
                print(f"  [ERROR] {year}: Could not load model for {period}: {e}")
                continue

        scaler, model = model_cache[period]

        # Classify
        try:
            result_df = classify_year(year, scaler, model)

            if result_df is not None:
                output_path = OUTPUT_DIR / f"classified_{year}.parquet"
                result_df.to_parquet(output_path, index=False)
                print(f"  {year}: Classified {len(result_df):,} documents (mean: {result_df['predicted_quality'].mean():.2f})")
            else:
                print(f"  [SKIP] {year}: No clean rows")
        except Exception as e:
            print(f"  [ERROR] {year}: {e}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Check status and classify new embeddings")
    parser.add_argument('--status', action='store_true', help='Only show status, do not classify')
    args = parser.parse_args()

    # Get and print status
    status = get_status()
    print_status(status)

    # Classify if not --status only
    if not args.status:
        classify_new(status)

        # Print updated status
        print("\n" + "=" * 60)
        print("UPDATED STATUS")
        print("=" * 60)
        status = get_status()
        print_status(status)


if __name__ == "__main__":
    main()
