"""
Export sampled documents from additional data sources as individual .txt files
for consumption by Meta's synthetic-data-kit.

Supports four sources: NYT (filtered), Economist, FT, Newswire.
After export, use run.py --skip-export --collection <name> to process.

Usage:
    # Export all datasets for 1950-1999
    python -m src.post_training.corpus.export_additional --period 1950_1999

    # Export only Economist and FT
    python -m src.post_training.corpus.export_additional --period 1950_1999 --dataset economist ft

    # Test with small sample
    python -m src.post_training.corpus.export_additional --period 1950_1999 --max-per-collection 50
"""

import os
import json
import argparse
from pathlib import Path

import pandas as pd

from src.post_training.config import PERIODS, get_paths


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "nyt_filtered": {
        "config_key": "nyt_filtered_dir",
        "description": "NYT filtered abstracts (500+ chars)",
    },
    "economist": {
        "config_key": "economist_dir",
        "description": "The Economist (OCR text)",
    },
    "ft": {
        "config_key": "ft_dir",
        "description": "Financial Times (cleaned text)",
    },
    "newswire": {
        "config_key": "newswire_dir",
        "description": "US Newswire articles (cleaned)",
    },
}


# ---------------------------------------------------------------------------
# Per-dataset loaders — each returns DataFrame with [text, source_id]
# ---------------------------------------------------------------------------

def _load_nyt_filtered(data_dir, start_year, end_year):
    """Load NYT filtered abstracts from yearly parquet files."""
    frames = []
    for year in range(max(start_year, 1851), min(end_year, 2017) + 1):
        path = data_dir / f"nyt_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["combined_text", "_id"])
        df.rename(columns={"combined_text": "text", "_id": "source_id"}, inplace=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["text", "source_id"])
    return pd.concat(frames, ignore_index=True)


def _load_economist(data_dir, start_year, end_year):
    """Load Economist articles from weekly parquet files."""
    all_files = sorted(data_dir.glob("economist_*.parquet"))
    matching = []
    for f in all_files:
        try:
            year = int(f.stem.split("_")[1].split("-")[0])
        except (IndexError, ValueError):
            continue
        if start_year <= year <= end_year:
            matching.append(f)

    if not matching:
        return pd.DataFrame(columns=["text", "source_id"])

    frames = []
    for i, f in enumerate(matching):
        try:
            df = pd.read_parquet(f, columns=["ocr_text", "article_id"])
            df.rename(columns={"ocr_text": "text", "article_id": "source_id"},
                      inplace=True)
            frames.append(df)
        except Exception as e:
            print(f"  Warning: Failed to read {f.name}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  Read {i + 1}/{len(matching)} Economist files...")

    if not frames:
        return pd.DataFrame(columns=["text", "source_id"])
    return pd.concat(frames, ignore_index=True)


def _load_ft(data_dir, start_year, end_year):
    """Load Financial Times articles from yearly parquet files."""
    frames = []
    for year in range(max(start_year, 1888), min(end_year, 2006) + 1):
        path = data_dir / f"{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["text_cleaned", "id"])
        df.rename(columns={"text_cleaned": "text", "id": "source_id"}, inplace=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["text", "source_id"])
    return pd.concat(frames, ignore_index=True)


def _load_newswire(data_dir, start_year, end_year):
    """Load Newswire articles from yearly JSON files."""
    frames = []
    for year in range(max(start_year, 1878), min(end_year, 1977) + 1):
        path = data_dir / f"{year}_data_clean.json"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Warning: Skipping {path.name} (corrupted JSON: {e})")
            continue
        records = []
        for idx, item in enumerate(data):
            text = item.get("cleaned_article", "")
            records.append({
                "text": text,
                "source_id": f"newswire_{year}_{idx}",
            })
        frames.append(pd.DataFrame(records))
    if not frames:
        return pd.DataFrame(columns=["text", "source_id"])
    return pd.concat(frames, ignore_index=True)


LOADERS = {
    "nyt_filtered": _load_nyt_filtered,
    "economist": _load_economist,
    "ft": _load_ft,
    "newswire": _load_newswire,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def export_additional(period, datasets=None, max_per_collection=10000,
                      min_text_length=200, seed=42):
    """Export additional datasets as parquet files for run_direct.py.

    Output structure:
        synthetic/input/{dataset_name}.parquet  (columns: doc_name, text)

    Steps per dataset:
    1. Load all documents for the period's year range
    2. Filter: drop null/empty text and text shorter than min_text_length
    3. Sample up to max_per_collection (or take all if smaller)
    4. Save as single parquet file (fast I/O vs 10K individual .txt files)
    """
    paths = get_paths(period)
    start_year, end_year = paths["start_year"], paths["end_year"]
    input_base = paths["synthetic_dir"] / "input"
    os.makedirs(input_base, exist_ok=True)

    if datasets is None:
        datasets = list(DATASETS.keys())

    print(f"Period: {period} ({start_year}-{end_year})")
    print(f"Max per collection: {max_per_collection:,}")
    print(f"Min text length: {min_text_length} chars")
    print(f"Datasets: {', '.join(datasets)}")

    summary = []

    for ds_name in datasets:
        ds_info = DATASETS[ds_name]
        data_dir = paths[ds_info["config_key"]]
        loader = LOADERS[ds_name]

        print(f"\n{'=' * 60}")
        print(f"Loading {ds_name}: {ds_info['description']}")
        print(f"  Source: {data_dir}")
        print(f"  Years: {start_year}-{end_year}")

        if not data_dir.exists():
            print(f"  Warning: Directory not found, skipping")
            summary.append((ds_name, 0, 0, 0, "skipped"))
            continue

        # Load
        pool = loader(data_dir, start_year, end_year)
        raw_count = len(pool)

        # Filter: non-null, non-empty, meets minimum length
        pool = pool[pool["text"].notna()]
        pool = pool[pool["text"].str.strip().str.len() >= min_text_length]
        filtered_count = len(pool)

        print(f"  Raw documents: {raw_count:,}")
        print(f"  After length filter (>={min_text_length} chars): {filtered_count:,}")

        if filtered_count == 0:
            print(f"  No documents passed filtering, skipping")
            summary.append((ds_name, raw_count, 0, 0, "empty"))
            continue

        # Sample
        if filtered_count > max_per_collection:
            sample = pool.sample(n=max_per_collection, random_state=seed)
            action = "sampled"
        else:
            sample = pool
            action = "all"
        n_selected = len(sample)

        # Build output dataframe with doc_name and text columns
        sample = sample.reset_index(drop=True)
        sample["doc_name"] = [f"doc_{i:05d}" for i in range(len(sample))]
        output_df = sample[["doc_name", "text"]]

        # Write single parquet file
        output_path = input_base / f"{ds_name}.parquet"
        output_df.to_parquet(output_path, index=False)

        print(f"  Written: {n_selected:,} documents ({action})")
        print(f"  Output: {output_path}")
        summary.append((ds_name, raw_count, filtered_count, n_selected, action))

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"Export Summary:")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<20} {'Raw':>10} {'Filtered':>10} {'Written':>10}  Action")
    print(f"{'-' * 70}")
    for ds_name, raw, filt, written, action in summary:
        print(f"{ds_name:<20} {raw:>10,} {filt:>10,} {written:>10,}  {action}")
    print(f"{'-' * 70}")
    total_written = sum(s[3] for s in summary)
    print(f"{'TOTAL':<20} {'':>10} {'':>10} {total_written:>10,}")
    print(f"\nOutput base: {input_base}")

    return input_base


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export additional data sources as .txt files for synthetic-data-kit"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()),
                        help="Time period to process")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        choices=list(DATASETS.keys()),
                        help="Specific datasets to export (default: all)")
    parser.add_argument("--max-per-collection", type=int, default=10000,
                        help="Max documents per dataset (default: 10000)")
    parser.add_argument("--min-text-length", type=int, default=200,
                        help="Minimum text length in characters (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    export_additional(
        args.period,
        datasets=args.dataset,
        max_per_collection=args.max_per_collection,
        min_text_length=args.min_text_length,
        seed=args.seed,
    )
