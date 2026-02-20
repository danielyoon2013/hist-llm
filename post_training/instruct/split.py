"""
Split filtered datasets and historical corpus into train/test JSONL files.

Reads all *_filtered.jsonl and hist_*.jsonl files from final/filtered/,
shuffles deterministically, and writes {name}_train.jsonl to final/train/
and {name}_test.jsonl to final/test/.

Pipeline order:
    filter.py (LAB filtering) → split.py (train/test) → nanochat training

Usage:
    # Split all datasets for a period (default 95/5)
    python -m src.post_training.instruct.split --period 1950_1999

    # Custom test ratio
    python -m src.post_training.instruct.split --period 1950_1999 --test-ratio 0.10

    # Only split specific datasets
    python -m src.post_training.instruct.split --period 1950_1999 --dataset smoltalk mmlu hist_corpus_qa

    # Dry run (show what would be created without writing)
    python -m src.post_training.instruct.split --period 1950_1999 --dry-run
"""

import argparse
import random
from pathlib import Path

from src.post_training.config import PERIODS, get_paths


SEED = 42
DEFAULT_TEST_RATIO = 0.05  # 95/5 split, consistent with SmolTalk's native ratio


def split_jsonl(input_path, train_path, test_path, test_ratio=DEFAULT_TEST_RATIO,
                seed=SEED, dry_run=False):
    """
    Split a JSONL file into train and test files.

    Args:
        input_path: Path to input JSONL file
        train_path: Path to write train split
        test_path: Path to write test split
        test_ratio: Fraction of data for test (default 0.05)
        seed: Random seed for reproducible shuffle
        dry_run: If True, compute sizes but don't write files

    Returns:
        (total, train_size, test_size) tuple
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    total = len(lines)
    if total == 0:
        return 0, 0, 0

    # Shuffle with deterministic seed
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    # Split: first test_size indices go to test, rest to train
    test_size = max(1, int(total * test_ratio))
    train_size = total - test_size
    test_indices = sorted(indices[:test_size])
    train_indices = sorted(indices[test_size:])

    if not dry_run:
        with open(train_path, "w", encoding="utf-8") as f:
            for idx in train_indices:
                f.write(lines[idx] if lines[idx].endswith("\n") else lines[idx] + "\n")

        with open(test_path, "w", encoding="utf-8") as f:
            for idx in test_indices:
                f.write(lines[idx] if lines[idx].endswith("\n") else lines[idx] + "\n")

    return total, train_size, test_size


def run_split(period, datasets=None, test_ratio=DEFAULT_TEST_RATIO, dry_run=False):
    """
    Split all filtered datasets + historical corpus for a period.

    Reads from final/filtered/:
      - *_filtered.jsonl  (LAB-filtered external datasets)
      - hist_*.jsonl       (per-collection historical corpus)

    Writes to final/train/ and final/test/:
      - train/{name}_train.jsonl
      - test/{name}_test.jsonl
    """
    paths = get_paths(period)
    filtered_dir = paths["final_filtered_dir"]
    train_dir = paths["final_train_dir"]
    test_dir = paths["final_test_dir"]

    if not filtered_dir.exists():
        print(f"Error: filtered/ directory not found at {filtered_dir}")
        return

    # Ensure output directories exist
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Collect files to split
    files_to_split = []

    # 1. Filtered external datasets (exclude hist_* to avoid overlap)
    for f in sorted(filtered_dir.glob("*_filtered.jsonl")):
        if f.name.startswith("hist_"):
            continue  # handled in section 2
        name = f.stem.replace("_filtered", "")
        files_to_split.append((name, f))

    # 2. Historical corpus (per-collection: hist_*.jsonl)
    for f in sorted(filtered_dir.glob("hist_*.jsonl")):
        name = f.stem  # e.g. "hist_economist"
        files_to_split.append((name, f))

    # Filter to requested datasets if specified
    if datasets:
        files_to_split = [(name, f) for name, f in files_to_split if name in datasets]

    if not files_to_split:
        print("No files found to split.")
        return

    action = "Would split" if dry_run else "Splitting"
    print(f"Period: {period}")
    print(f"Test ratio: {test_ratio*100:.0f}%")
    print(f"Seed: {SEED}")
    print(f"Input:  {filtered_dir}")
    print(f"Train:  {train_dir}")
    print(f"Test:   {test_dir}")
    if dry_run:
        print("DRY RUN — no files will be written")
    print()

    results = []
    for name, filepath in files_to_split:
        train_path = train_dir / f"{name}_train.jsonl"
        test_path = test_dir / f"{name}_test.jsonl"

        total, train_size, test_size = split_jsonl(
            filepath, train_path, test_path,
            test_ratio=test_ratio, dry_run=dry_run,
        )

        results.append((name, total, train_size, test_size))
        status = "(dry run)" if dry_run else "done"
        print(f"  {name}: {total:,} -> {train_size:,} train + {test_size:,} test  {status}",
              flush=True)

    # Summary
    total_all = sum(r[1] for r in results)
    total_train = sum(r[2] for r in results)
    total_test = sum(r[3] for r in results)

    print(f"\n{'='*72}")
    print(f"{'Dataset':<37} {'Total':>10} {'Train':>10} {'Test':>10}")
    print(f"{'-'*72}")
    for name, total, train, test in results:
        print(f"{name:<37} {total:>10,} {train:>10,} {test:>10,}")
    print(f"{'-'*72}")
    print(f"{'TOTAL':<37} {total_all:>10,} {total_train:>10,} {total_test:>10,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split filtered datasets into train/test for nanochat training"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="Specific datasets to split (default: all)")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO,
                        help=f"Fraction for test split (default: {DEFAULT_TEST_RATIO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be created without writing files")
    args = parser.parse_args()

    run_split(
        args.period,
        datasets=args.dataset,
        test_ratio=args.test_ratio,
        dry_run=args.dry_run,
    )
