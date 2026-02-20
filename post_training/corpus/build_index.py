"""
Build a metadata index joining quality scores and collection labels
for all documents in a period.

Reads from:
  - D:/English_Classified/classified_{year}.parquet  (identifier, predicted_quality)
  - D:/English/{year}/subset_*.parquet               (identifier, collection)

Writes to:
  training_data/{period}/document_metadata.parquet

Usage:
    python -m src.post_training.corpus.build_index --period 1950_1999
"""

import os
import argparse
from glob import glob
from pathlib import Path

import pandas as pd

from src.post_training.config import PERIODS, get_paths


def build_index(period):
    """Build a metadata index for all documents in the given period."""
    paths = get_paths(period)
    start_year, end_year = paths["start_year"], paths["end_year"]
    raw_root = paths["raw_data_root"]
    classified_root = paths["classified_root"]
    output_path = paths["metadata_index"]

    frames = []
    for year in range(start_year, end_year + 1):
        # 1. Quality scores (small file, fast)
        classified_path = classified_root / f"classified_{year}.parquet"
        if not classified_path.exists():
            print(f"  Warning: {classified_path} not found, skipping year {year}")
            continue

        classified = pd.read_parquet(
            classified_path, columns=["identifier", "predicted_quality"]
        )

        # 2. Collection info from raw subsets (column projection, no text)
        year_dir = raw_root / str(year)
        if not year_dir.exists():
            print(f"  Warning: {year_dir} not found, skipping year {year}")
            continue

        subset_frames = []
        for f in sorted(glob(str(year_dir / "subset_*.parquet"))):
            df = pd.read_parquet(f, columns=["identifier", "collection"])
            df["subset_file"] = Path(f).name
            subset_frames.append(df)

        if not subset_frames:
            print(f"  Warning: no subset files in {year_dir}, skipping")
            continue

        raw = pd.concat(subset_frames, ignore_index=True)

        # 3. Join on identifier
        merged = classified.merge(raw, on="identifier", how="inner")
        merged["year"] = year
        frames.append(merged)

        print(f"  {year}: {len(merged):,} docs "
              f"({merged['collection'].nunique()} collections)")

    if not frames:
        print("No data found. Check that D: drive is accessible.")
        return

    index = pd.concat(frames, ignore_index=True)

    # Write output
    os.makedirs(output_path.parent, exist_ok=True)
    index.to_parquet(output_path, index=False)

    print(f"\nMetadata index built:")
    print(f"  Total documents: {len(index):,}")
    print(f"  Years: {start_year}-{end_year}")
    print(f"  Collections: {index['collection'].nunique()}")
    print(f"  Quality range: {index['predicted_quality'].min():.2f} - "
          f"{index['predicted_quality'].max():.2f}")
    print(f"  Output: {output_path}")

    # Print collection breakdown
    print(f"\nCollection breakdown:")
    counts = index["collection"].value_counts()
    for col, count in counts.items():
        pct = 100 * count / len(index)
        print(f"  {col}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build document metadata index from D: drive sources"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    args = parser.parse_args()

    paths = get_paths(args.period)
    print(f"Period: {args.period} ({paths['start_year']}-{paths['end_year']})")
    print(f"Raw data: {paths['raw_data_root']}")
    print(f"Classified: {paths['classified_root']}")
    print(f"Output: {paths['metadata_index']}")
    print()

    build_index(args.period)
