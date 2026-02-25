"""
Assemble synthetic data into nanochat-ready training files.

Reads per-generator JSONL from quality/deduped/ (or final/filtered/),
merges them into per-collection hist_{collection}_train.jsonl files
matching the naming in speedrun_hist_llm.sh.

Usage:
    python -m src.post_training.assemble --period 1900_1949
    python -m src.post_training.assemble --period 1900_1949 --dry-run
    python -m src.post_training.assemble --period 1900_1949 --source deduped
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from src.post_training.config import PERIODS, get_paths
from src.post_training.utils import read_jsonl, write_jsonl


SEED = 42
DEFAULT_TEST_RATIO = 0.05


def find_input_dir(paths):
    """Find the best available input directory in priority order.

    Priority: final/filtered > quality/deduped > quality/validated > synthetic/by_generator
    """
    candidates = [
        paths["final_filtered_dir"],
        paths["synthetic_dir"].parent / "quality" / "deduped",
        paths["synthetic_dir"].parent / "quality" / "validated",
        paths["synthetic_dir"] / "by_generator",
    ]
    for d in candidates:
        if d.exists() and any(d.glob("*.jsonl")):
            return d
    return None


def load_all_generators(input_dir):
    """Load all generator JSONL files from a directory.

    Returns dict mapping generator_name -> list of conversations.
    """
    data = {}
    for jsonl_file in sorted(Path(input_dir).glob("*.jsonl")):
        name = jsonl_file.stem
        conversations = read_jsonl(str(jsonl_file))
        if conversations:
            data[name] = conversations
    return data


def merge_generators(generator_data):
    """Merge all generator outputs into a single list, shuffled.

    Returns list of conversations.
    """
    all_convs = []
    for gen_name, convs in generator_data.items():
        all_convs.extend(convs)

    rng = random.Random(SEED)
    rng.shuffle(all_convs)
    return all_convs


def split_train_test(conversations, test_ratio=DEFAULT_TEST_RATIO):
    """Split conversations into train/test with deterministic shuffle.

    Returns (train_list, test_list).
    """
    rng = random.Random(SEED)
    indices = list(range(len(conversations)))
    rng.shuffle(indices)

    test_size = max(1, int(len(conversations) * test_ratio))
    test_indices = set(indices[:test_size])

    train = [conversations[i] for i in range(len(conversations)) if i not in test_indices]
    test = [conversations[i] for i in range(len(conversations)) if i in test_indices]
    return train, test


def assemble(period, source=None, test_ratio=DEFAULT_TEST_RATIO, dry_run=False):
    """Assemble synthetic data for a period.

    Creates a single merged file for all synthetic generators in
    final/train/ and final/test/.

    Args:
        period: Period key
        source: Override input directory ("deduped", "validated", "raw", or a path)
        test_ratio: Fraction for test split
        dry_run: Show plan without writing

    Returns:
        dict with assembly stats
    """
    paths = get_paths(period)
    train_dir = paths["final_train_dir"]
    test_dir = paths["final_test_dir"]

    # Determine input
    if source == "deduped":
        input_dir = paths["synthetic_dir"].parent / "quality" / "deduped"
    elif source == "validated":
        input_dir = paths["synthetic_dir"].parent / "quality" / "validated"
    elif source == "raw":
        input_dir = paths["synthetic_dir"] / "by_generator"
    elif source:
        input_dir = Path(source)
    else:
        input_dir = find_input_dir(paths)

    if input_dir is None or not input_dir.exists():
        print(f"ERROR: No input directory found. Run generate.py and process.py first.")
        return {}

    print(f"Period: {period}")
    print(f"Input:  {input_dir}")
    print(f"Train:  {train_dir}")
    print(f"Test:   {test_dir}")
    if dry_run:
        print("DRY RUN — no files will be written")

    # Load all generators
    generator_data = load_all_generators(input_dir)
    if not generator_data:
        print("No generator data found!")
        return {}

    print(f"\nLoaded {len(generator_data)} generator files:")
    for name, convs in generator_data.items():
        print(f"  {name}: {len(convs):,} conversations")

    # Merge into single synthetic dataset
    all_convs = merge_generators(generator_data)
    total = len(all_convs)
    print(f"\nTotal merged: {total:,} conversations")

    # Split
    train, test = split_train_test(all_convs, test_ratio=test_ratio)
    print(f"Train: {len(train):,} | Test: {len(test):,}")

    # Write output as hist_synthetic_train.jsonl / hist_synthetic_test.jsonl
    # These can be added to speedrun_hist_llm.sh's TRAIN_FILES array
    if not dry_run:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        train_path = train_dir / "hist_synthetic_train.jsonl"
        test_path = test_dir / "hist_synthetic_test.jsonl"

        write_jsonl(train, str(train_path), validate=False)
        write_jsonl(test, str(test_path), validate=False)

        print(f"\nWritten:")
        print(f"  {train_path}")
        print(f"  {test_path}")

    stats = {
        "input_dir": str(input_dir),
        "generators": {name: len(convs) for name, convs in generator_data.items()},
        "total": total,
        "train": len(train),
        "test": len(test),
    }

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Generator':<30} {'Count':>10}")
    print(f"{'-'*60}")
    for name, count in sorted(stats["generators"].items()):
        print(f"{name:<30} {count:>10,}")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<30} {total:>10,}")
    print(f"{'Train':<30} {len(train):>10,}")
    print(f"{'Test':<30} {len(test):>10,}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble synthetic data into nanochat training files"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--source", type=str, default=None,
                        choices=["deduped", "validated", "raw"],
                        help="Input source (default: auto-detect best available)")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO,
                        help=f"Test split ratio (default: {DEFAULT_TEST_RATIO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without writing files")
    args = parser.parse_args()

    assemble(args.period, source=args.source, test_ratio=args.test_ratio,
             dry_run=args.dry_run)
