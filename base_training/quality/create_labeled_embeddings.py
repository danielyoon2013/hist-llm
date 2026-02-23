r"""
Create labeled embeddings by joining GPT quality labels with BGE embeddings.

Joins the labeled sample data (from Label_Data.ipynb) with the corresponding
BGE embeddings to produce training data for Ridge models.

Input:
    D:\hist_LLM\processing\label_data\labeled_data_{period}.parquet
        Columns: [text, original_index, year, score]
    D:\hist_LLM\corpus\embeddings\embeddings_{year}.parquet
        Columns: [original_index, embedding, ...]

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

# --- CONFIG ---
LABEL_DATA_DIR = Path(r"D:\hist_LLM\processing\label_data")
EMBEDDING_DIR = Path(r"D:\hist_LLM\corpus\embeddings")
OUTPUT_DIR = Path(r"D:\hist_LLM\processing\labeled_embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERIODS = [
    "1678_1700", "1701_1725", "1726_1750", "1751_1775",
    "1776_1800", "1801_1825", "1826_1850", "1851_1875",
    "1876_1900", "1901_1925", "1926_1950", "1951_1975",
    "1976_2000", "2001_2023",
]


def process_period(period: str):
    """Create labeled embeddings for a single period."""
    label_path = LABEL_DATA_DIR / f"labeled_data_{period}.parquet"
    output_path = OUTPUT_DIR / f"embeddings_bge_{period}.parquet"

    if not label_path.exists():
        print(f"  [SKIP] No labeled data: {label_path.name}")
        return

    label_df = pd.read_parquet(label_path, columns=["original_index", "year", "score"])
    print(f"  Loaded {len(label_df)} labeled samples")

    # Load embeddings for each year in this period, merge with labels
    years = sorted(label_df["year"].unique())
    results = []

    for year in years:
        emb_path = EMBEDDING_DIR / f"embeddings_{year}.parquet"
        if not emb_path.exists():
            print(f"  [WARN] No embeddings for year {year}")
            continue

        year_labels = label_df[label_df["year"] == year].copy()
        year_embs = pd.read_parquet(emb_path, columns=["original_index", "embedding"])

        # Deduplicate embeddings (some years have dupes)
        year_embs = year_embs.drop_duplicates(subset="original_index", keep="first")

        # Join on original_index (both as strings)
        year_labels["original_index"] = year_labels["original_index"].astype(str)
        year_embs["original_index"] = year_embs["original_index"].astype(str)

        merged = year_labels.merge(year_embs, on="original_index", how="inner")

        # Normalize embedding dtype (some years are float16, others float32)
        merged["embedding"] = merged["embedding"].apply(lambda e: e.astype(np.float32))

        results.append(merged[["original_index", "score", "embedding"]])

    if not results:
        print(f"  [WARN] No embeddings matched for {period}")
        return

    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df.rename(columns={"score": "labels"})
    result_df.to_parquet(output_path, index=False)

    matched = len(result_df)
    total = len(label_df)
    print(f"  Saved {matched}/{total} ({100*matched/total:.1f}%) to {output_path.name}")

    # Print label distribution
    for score in range(1, 6):
        count = (result_df["labels"] == score).sum()
        print(f"    Score {score}: {count} ({100*count/matched:.1f}%)")


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
