"""
Check embedding/classification status and classify English corpus embeddings.

Only classifies documents that pass heuristic cleaning masks (from Clean_Data.ipynb).
This ensures garbled OCR, tables/lists, and non-English text are excluded before
quality scoring.

Usage:
    python check_and_classify.py                    # Classify unclassified years
    python check_and_classify.py --status           # Only show status
    python check_and_classify.py --reclassify       # Reclassify all (overwrite existing)
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
RAW_DIR = Path(r"D:\hist_LLM\corpus\raw")
EMBEDDINGS_DIR = Path(r"D:\hist_LLM\corpus\embeddings")
MASKS_DIR = Path(r"D:\hist_LLM\corpus\cleaning_masks")
MODELS_DIR = Path(r"D:\hist_LLM\processing\quality_models")
OUTPUT_DIR = Path(r"D:\hist_LLM\corpus\classified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Separate destination for the --all-docs audit pass (scores kept + dropped).
# Kept distinct so the production classified/ output is never overwritten.
ALL_OUTPUT_DIR = Path(r"D:\hist_LLM\corpus\classified_all")

# 25-year periods
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
    all_years = sorted([int(d.name) for d in RAW_DIR.iterdir()
                        if d.is_dir() and d.name.isdigit()])

    emb_years = set()
    for f in EMBEDDINGS_DIR.glob('embeddings_*.parquet'):
        try:
            year = int(f.stem.split('_')[1])
            emb_years.add(year)
        except:
            pass

    classified_years = set()
    for f in OUTPUT_DIR.glob('classified_*.parquet'):
        try:
            year = int(f.stem.split('_')[1])
            classified_years.add(year)
        except:
            pass

    return {
        'all_years': set(all_years),
        'has_embedding': emb_years,
        'classified': classified_years,
        'ready_to_classify': emb_years - classified_years,
        'missing_embedding': set(all_years) - emb_years,
    }


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


def load_clean_indices(year: int) -> set:
    """Load the set of clean original_index values from cleaning masks."""
    mask_dir = MASKS_DIR / str(year)
    if not mask_dir.exists():
        return None

    clean_indices = set()
    for mask_path in mask_dir.glob("*_mask.parquet"):
        # Mask files store row indices into the subset file.
        # We need to map these back to original_index (identifier).
        # The mask file name is {subset}_mask.parquet
        subset_name = mask_path.stem.replace("_mask", "") + ".parquet"
        raw_path = RAW_DIR / str(year) / subset_name
        if not raw_path.exists():
            continue

        mask_df = pd.read_parquet(mask_path)
        clean_rows = set(mask_df["original_index"].tolist())

        # Read identifiers from raw file, pick only clean rows
        raw_df = pd.read_parquet(raw_path, columns=["identifier"])
        for row_idx in clean_rows:
            if row_idx < len(raw_df):
                clean_indices.add(str(raw_df.iloc[row_idx]["identifier"]))

    return clean_indices


def get_token_word_counts_english(year: int, identifiers: list) -> tuple:
    """Get token_count and word_count from English raw data."""
    raw_dir = RAW_DIR / str(year)
    if not raw_dir.exists():
        return {}, {}

    token_counts = {}
    word_counts = {}
    id_set = set(str(i) for i in identifiers)

    for subset_path in raw_dir.glob("subset_*.parquet"):
        try:
            raw_df = pd.read_parquet(subset_path, columns=['identifier', 'token_count', 'word_count'])
            raw_df['identifier'] = raw_df['identifier'].astype(str)
            hit = raw_df[raw_df['identifier'].isin(id_set)]
            # vectorized join (avoids per-row iterrows; matters when scoring ALL docs)
            token_counts.update(dict(zip(hit['identifier'], hit['token_count'])))
            word_counts.update(dict(zip(hit['identifier'], hit['word_count'])))
        except:
            pass

    return token_counts, word_counts


def classify_year(year: int, scaler, model, all_docs: bool = False) -> pd.DataFrame:
    """Classify documents for a single year.

    Default: only embedding rows whose original_index passes the cleaning mask
    (is_clean=True for all output rows). With all_docs=True: score EVERY doc,
    skipping the mask filter, and set is_clean per the mask (True=kept, False=
    heuristically dropped). Same output schema either way.
    """
    emb_path = EMBEDDINGS_DIR / f"embeddings_{year}.parquet"
    emb_df = pd.read_parquet(emb_path)

    if len(emb_df) == 0:
        return None

    # Load cleaning mask. Default path filters to clean rows; all_docs keeps all
    # rows and instead uses the mask to label is_clean per row.
    clean_indices = load_clean_indices(year)
    if all_docs:
        ids_series = emb_df['original_index'].astype(str)
        if clean_indices is not None:
            is_clean_col = ids_series.isin(clean_indices).tolist()
        else:
            # No mask for this year => heuristic never applied => treat as clean
            is_clean_col = [True] * len(emb_df)
    else:
        if clean_indices is not None:
            emb_df = emb_df[emb_df['original_index'].astype(str).isin(clean_indices)]
            if len(emb_df) == 0:
                return None
        is_clean_col = True

    # Predict quality scores
    X = np.stack(emb_df['embedding'].values)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Get token_count and word_count from raw data
    identifiers = emb_df['original_index'].astype(str).tolist()
    token_counts, word_counts = get_token_word_counts_english(year, identifiers)

    result_df = pd.DataFrame({
        'identifier': identifiers,
        'predicted_quality': predictions,
        'is_clean': is_clean_col,
        'token_count': [token_counts.get(str(i), None) for i in identifiers],
        'word_count': [word_counts.get(str(i), None) for i in identifiers]
    })

    return result_df


def classify_english(reclassify: bool = False, all_docs: bool = False, years: list = None):
    """Classify English corpus embeddings.

    all_docs=False (default): clean rows only -> D:\\hist_LLM\\corpus\\classified\\
    all_docs=True: ALL docs (kept + dropped), is_clean labeled per mask
                   -> D:\\hist_LLM\\corpus\\classified_all\\ (separate folder; the
                   production classified\\ output is never touched).
    years: optional explicit list of years to process (for small test runs).
    """
    status = get_status()
    print_status(status)

    # all_docs writes to a separate sibling folder so nothing is overwritten
    out_dir = ALL_OUTPUT_DIR if all_docs else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if years:
        ready_years = sorted(y for y in years if y in status['has_embedding'])
        print(f"Processing {len(ready_years)} requested year(s): {ready_years}")
    elif all_docs or reclassify:
        # score every year that has embeddings (fresh output dir for all_docs)
        ready_years = sorted(status['has_embedding'])
        print(f"{'ALL-DOCS scoring' if all_docs else 'Reclassifying'} {len(ready_years)} years...")
    else:
        ready_years = sorted(status['ready_to_classify'])
        if not ready_years:
            print("No new years to classify.")
            return

    print(f"Classifying {len(ready_years)} years -> {out_dir} (all_docs={all_docs})")
    model_cache = {}

    for year in ready_years:
        period = get_period_for_year(year)
        if not period:
            continue

        if period not in model_cache:
            try:
                model_cache[period] = load_models(period)
            except Exception as e:
                print(f"  [ERROR] {year}: Could not load model for {period}: {e}")
                continue

        scaler, model = model_cache[period]

        try:
            result_df = classify_year(year, scaler, model, all_docs=all_docs)
            if result_df is not None:
                output_path = out_dir / f"classified_{year}.parquet"
                result_df.to_parquet(output_path, index=False)
                if all_docs:
                    nkept = int(result_df['is_clean'].sum())
                    print(f"  {year}: {len(result_df):,} docs ({nkept:,} kept / {len(result_df)-nkept:,} dropped) "
                          f"mean q={result_df['predicted_quality'].mean():.2f}")
                else:
                    print(f"  {year}: {len(result_df):,} docs (mean: {result_df['predicted_quality'].mean():.2f})")
            else:
                print(f"  [SKIP] {year}: No embeddings")
        except Exception as e:
            print(f"  [ERROR] {year}: {e}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Classify embeddings with Ridge quality models")
    parser.add_argument('--status', action='store_true', help='Only show status')
    parser.add_argument('--reclassify', action='store_true', help='Reclassify all (overwrite existing)')
    parser.add_argument('--all-docs', action='store_true',
                        help='Score ALL docs incl. heuristically-dropped ones (is_clean labeled per '
                             'mask); writes to corpus/classified_all/ without touching classified/')
    parser.add_argument('--years', type=str, default=None,
                        help='Comma-separated years to process (e.g. 1850,1950,2000) for a small test run')
    args = parser.parse_args()

    if args.status:
        status = get_status()
        print_status(status)
        return

    years = [int(y) for y in args.years.split(',')] if args.years else None
    classify_english(reclassify=args.reclassify, all_docs=args.all_docs, years=years)


if __name__ == "__main__":
    main()
