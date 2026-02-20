"""
Create labeled embeddings by joining GPT quality labels with BGE embeddings.

This script joins the labeled sample data (from Label_Data.ipynb) with
the corresponding BGE embeddings to produce training data for Ridge models.
Handles both English corpus and additional news data embeddings.

Input:
    D:\hist_LLM\processing\label_data\labeled_data_{period}.parquet
        Columns: [text, original_index, year, source, score]
    D:\hist_LLM\corpus\embeddings\embeddings_{year}.parquet  (English)
    D:\hist_LLM\additional_data\embeddings\{collection}\embeddings_{year}.parquet  (Additional)

Output:
    D:\hist_LLM\processing\labeled_embeddings\embeddings_bge_{period}.parquet
        Columns: [original_index, labels, embedding]

Usage:
    python create_labeled_embeddings.py                    # All periods
    python create_labeled_embeddings.py --period 1901_1925 # Single period
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
LABEL_DATA_DIR = Path(r"D:\hist_LLM\processing\label_data")
ENGLISH_EMB_DIR = Path(r"D:\hist_LLM\corpus\embeddings")
ADDITIONAL_EMB_DIR = Path(r"D:\hist_LLM\additional_data\embeddings")
OUTPUT_DIR = Path(r"D:\hist_LLM\processing\labeled_embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Map source names to embedding directories
ADDITIONAL_COLLECTIONS = ["nyt", "economist", "ft", "newswire"]

PERIODS = [
    "1678_1700", "1701_1725", "1726_1750", "1751_1775",
    "1776_1800", "1801_1825", "1826_1850", "1851_1875",
    "1876_1900", "1901_1925", "1926_1950", "1951_1975",
    "1976_2000", "2001_2023",
]


def load_embeddings_for_year(year: int, source: str) -> dict:
    """Load embeddings and return {original_index: embedding} lookup."""
    if source == "english":
        path = ENGLISH_EMB_DIR / f"embeddings_{year}.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path, columns=["original_index", "embedding"])
        return dict(zip(df["original_index"].astype(str), df["embedding"]))
    elif source in ADDITIONAL_COLLECTIONS:
        path = ADDITIONAL_EMB_DIR / source / f"embeddings_{year}.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path, columns=["original_index", "embedding"])
        return dict(zip(df["original_index"].astype(str), df["embedding"]))
    return {}


def process_period(period: str):
    """Create labeled embeddings for a single period."""
    label_path = LABEL_DATA_DIR / f"labeled_data_{period}.parquet"
    output_path = OUTPUT_DIR / f"embeddings_bge_{period}.parquet"

    if not label_path.exists():
        print(f"  [SKIP] No labeled data: {label_path.name}")
        return

    # Load labeled data
    label_df = pd.read_parquet(label_path)
    print(f"  Loaded {len(label_df)} labeled samples")

    # Group by (year, source) for efficient embedding loading
    results = []
    emb_cache = {}

    for _, row in tqdm(label_df.iterrows(), total=len(label_df), desc=f"  Joining", leave=False):
        year = int(row["year"])
        source = row["source"]
        orig_idx = str(row["original_index"])
        score = row["score"]

        # Cache embeddings per (year, source)
        cache_key = (year, source)
        if cache_key not in emb_cache:
            emb_cache[cache_key] = load_embeddings_for_year(year, source)

        emb_lookup = emb_cache[cache_key]
        if orig_idx in emb_lookup:
            results.append({
                "original_index": orig_idx,
                "labels": score,
                "embedding": emb_lookup[orig_idx],
            })

    if not results:
        print(f"  [WARN] No embeddings matched for {period}")
        return

    result_df = pd.DataFrame(results)
    result_df.to_parquet(output_path, index=False)

    matched = len(results)
    total = len(label_df)
    print(f"  Saved {matched}/{total} ({100*matched/total:.1f}%) to {output_path.name}")

    # Print label distribution
    labels = result_df["labels"]
    for score in range(1, 6):
        count = (labels == score).sum()
        print(f"    Score {score}: {count} ({100*count/len(labels):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Create labeled embeddings for Ridge training")
    parser.add_argument("--period", type=str, help="Single period to process")
    args = parser.parse_args()

    periods = [args.period] if args.period else PERIODS

    print("Creating labeled embeddings...")
    for period in periods:
        print(f"\n{'='*50}")
        print(f"Period: {period}")
        print(f"{'='*50}")
        process_period(period)

    print("\nDone!")


if __name__ == "__main__":
    main()
