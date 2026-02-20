"""
Export sampled documents from parquet shards as individual .txt files
for consumption by Meta's synthetic-data-kit.

Exports documents per-collection with quality filtering:
  - If collection has > max_per_collection docs: sample max_per_collection
  - If collection has <= max_per_collection docs: take all

Requires the metadata index to be built first via build_index.py.

Usage:
    # Test with small sample (2 docs per collection)
    python -m src.post_training.corpus.export --period 1950_1999 --max-per-collection 2

    # Production (10K docs per collection, top 50% quality)
    python -m src.post_training.corpus.export --period 1950_1999

    # Custom settings
    python -m src.post_training.corpus.export --period 1950_1999 --max-per-collection 5000 --quality-percentile 75
"""

import os
import re
import argparse

import pandas as pd

from src.post_training.config import PERIODS, get_paths


def sanitize_collection_name(name):
    """Convert collection name to valid folder name."""
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r'[^\w\-]', '_', name)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')


def export_documents_stratified(period, max_per_collection=10000,
                                quality_percentile=50, seed=42):
    """Export documents per collection into separate folders.

    Output structure:
        synthetic/input/{collection}/doc_00000.txt, doc_00001.txt, ...

    Logic per collection:
      - If docs > max_per_collection: sample max_per_collection randomly
      - If docs <= max_per_collection: take all docs

    Steps:
    1. Load metadata index (must be built first via build_index.py)
    2. Apply quality floor (top N percentile)
    3. For each collection: sample max_per_collection OR take all if smaller
    4. Retrieve text from raw parquets
    5. Write .txt files to per-collection folders
    """
    paths = get_paths(period)
    input_base = paths["synthetic_dir"] / "input"
    metadata_path = paths["metadata_index"]
    raw_root = paths["raw_data_root"]

    if not metadata_path.exists():
        print(f"Metadata index not found: {metadata_path}")
        print("Run: python -m src.post_training.corpus.build_index "
              f"--period {period}")
        return None

    # 1. Load metadata
    print(f"Loading metadata index: {metadata_path}")
    meta = pd.read_parquet(metadata_path)
    print(f"  Total documents: {len(meta):,}")

    # 2. Apply quality floor
    threshold = meta["predicted_quality"].quantile(quality_percentile / 100)
    meta = meta[meta["predicted_quality"] >= threshold]
    print(f"  After quality filter (>= {threshold:.2f}, "
          f"top {100 - quality_percentile}%): {len(meta):,}")

    # 3. Per-collection sampling: sample if > max, else take all
    collections = sorted(meta["collection"].unique())
    print(f"\n  Processing {len(collections)} collections "
          f"(max {max_per_collection:,} per collection):\n")

    summary = []
    collection_samples = {}  # collection -> sampled dataframe

    for collection in collections:
        coll_df = meta[meta["collection"] == collection]
        available = len(coll_df)

        if available > max_per_collection:
            sample = coll_df.sample(n=max_per_collection, random_state=seed)
            action = "sampled"
            n = max_per_collection
        else:
            sample = coll_df
            action = "all"
            n = available

        collection_samples[collection] = sample
        summary.append((collection, available, n, action))
        print(f"    {collection}: {n:,} / {available:,} ({action})")

    total_sampled = sum(len(df) for df in collection_samples.values())
    print(f"\n  Total selected: {total_sampled:,}")

    # 4. Retrieve text from raw parquets (batch by year/subset for efficiency)
    print(f"\nRetrieving text from {raw_root}...")

    # Combine all samples to batch the parquet reads
    all_sampled = pd.concat(collection_samples.values(), ignore_index=True)
    texts = {}  # identifier -> (text, collection)

    for (year, subset_file), group_df in all_sampled.groupby(["year", "subset_file"]):
        parquet_path = raw_root / str(year) / subset_file
        if not parquet_path.exists():
            print(f"  Warning: {parquet_path} not found, skipping")
            continue
        needed_ids = set(group_df["identifier"])
        raw_df = pd.read_parquet(parquet_path, columns=["identifier", "text"])
        matched = raw_df[raw_df["identifier"].isin(needed_ids)]

        # Map identifier to collection
        id_to_collection = dict(zip(group_df["identifier"], group_df["collection"]))

        for _, row in matched.iterrows():
            if pd.notna(row["text"]) and row["text"].strip():
                texts[row["identifier"]] = (row["text"], id_to_collection[row["identifier"]])

    print(f"  Retrieved {len(texts):,} / {total_sampled:,} documents")

    # 5. Write .txt files to per-collection folders
    print(f"\nWriting to {input_base}/...")

    # Group by collection and write
    collection_counts = {}
    for collection in collections:
        folder_name = sanitize_collection_name(collection)
        coll_dir = input_base / folder_name
        os.makedirs(coll_dir, exist_ok=True)

        # Clear existing files
        for f in os.listdir(coll_dir):
            if f.endswith(".txt"):
                os.remove(coll_dir / f)

        collection_counts[collection] = 0

    # Write files
    doc_counts = {c: 0 for c in collections}
    for identifier, (text, collection) in texts.items():
        folder_name = sanitize_collection_name(collection)
        coll_dir = input_base / folder_name
        doc_idx = doc_counts[collection]
        filepath = coll_dir / f"doc_{doc_idx:05d}.txt"
        filepath.write_text(text, encoding="utf-8")
        doc_counts[collection] += 1

    total_written = sum(doc_counts.values())
    print(f"\nExported {total_written:,} documents")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"Summary by collection:")
    print(f"{'='*70}")
    print(f"{'Collection':<30} {'Available':>10} {'Selected':>10} {'Written':>10}")
    print(f"{'-'*70}")
    for coll, avail, sel, action in summary:
        written = doc_counts.get(coll, 0)
        print(f"{coll:<30} {avail:>10,} {sel:>10,} {written:>10,}")
    print(f"{'-'*70}")
    print(f"{'TOTAL':<30} {sum(s[1] for s in summary):>10,} "
          f"{sum(s[2] for s in summary):>10,} {total_written:>10,}")
    print(f"\nOutput: {input_base}")

    return input_base


def get_collection_folders(period):
    """Get list of collection folders in synthetic/input/."""
    paths = get_paths(period)
    input_base = paths["synthetic_dir"] / "input"
    if not input_base.exists():
        return []
    return [d.name for d in input_base.iterdir()
            if d.is_dir() and not d.name.startswith('_')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export sampled documents as .txt files for synthetic-data-kit"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys())),
    parser.add_argument("--max-per-collection", type=int, default=10000,
                        help="Max documents per collection (default: 10000)")
    parser.add_argument("--quality-percentile", type=int, default=50,
                        help="Quality floor percentile (default: 50 = top 50%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    paths = get_paths(args.period)
    print(f"Period: {args.period} ({paths['start_year']}-{paths['end_year']})")

    export_documents_stratified(
        args.period,
        max_per_collection=args.max_per_collection,
        quality_percentile=args.quality_percentile,
        seed=args.seed,
    )
