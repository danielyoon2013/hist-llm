"""
Assemble synthetic data into nanochat-ready training files.

Reads per-generator JSONL from quality/deduped/ (or by_generator/),
produces three output files:
  1. hist_synthetic_midtrain.jsonl  — all examples (1M target)
  2. hist_synthetic_sft.jsonl       — proportional subsample (10K target)
  3. hist_synthetic_test.jsonl      — 5% holdout for training-loss monitoring

Usage:
    python -m src.post_training.assemble --period 1900_1949
    python -m src.post_training.assemble --period 1900_1949 --sft-size 10000
    python -m src.post_training.assemble --period 1900_1949 --dry-run
"""

import os
import random
import argparse
from pathlib import Path
from collections import defaultdict

from src.post_training.config import PERIODS, get_paths, DEFAULT_SFT_SIZE, DEFAULT_TEST_RATIO
from src.post_training.utils import read_jsonl, write_jsonl


SEED = 42


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


def proportional_subsample(generator_data, target_size, exclude_prefixes=(), seed=SEED):
    """Subsample proportionally from each generator to hit target_size.

    Args:
        generator_data: dict of {gen_name: [conversations]}
        target_size: total examples to sample
        exclude_prefixes: generator name prefixes to skip (e.g. train-only gens)
        seed: random seed

    Returns:
        list of sampled conversations (shuffled)
    """
    rng = random.Random(seed)

    # Filter to eligible generators
    eligible = {
        name: convs for name, convs in generator_data.items()
        if not any(name.startswith(p) for p in exclude_prefixes)
    }

    total_eligible = sum(len(c) for c in eligible.values())
    if total_eligible == 0:
        return []

    # Compute per-generator sample sizes (proportional)
    samples = []
    remaining = target_size
    gen_items = sorted(eligible.items())  # deterministic order

    for i, (name, convs) in enumerate(gen_items):
        if i == len(gen_items) - 1:
            # Last generator gets the remainder to hit exact target
            n = remaining
        else:
            n = round(target_size * len(convs) / total_eligible)
        n = min(n, len(convs), remaining)
        remaining -= n

        if n > 0:
            sampled = rng.sample(convs, n)
            samples.extend(sampled)

    rng.shuffle(samples)
    return samples


def assemble(period, source=None, test_ratio=DEFAULT_TEST_RATIO,
             sft_size=DEFAULT_SFT_SIZE, dry_run=False):
    """Assemble synthetic data for a period.

    Produces three output files in final/:
      - hist_synthetic_midtrain.jsonl  (all examples for mid-training)
      - hist_synthetic_sft.jsonl       (proportional subsample for SFT)
      - hist_synthetic_test.jsonl      (holdout for monitoring)

    Args:
        period: Period key
        source: Override input directory ("deduped", "validated", "raw", or a path)
        test_ratio: Fraction for test split (monitoring only)
        sft_size: Number of SFT examples (proportional subsample)
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
        print(f"ERROR: No input directory found. Run generate.py first.")
        return {}

    print(f"Period: {period}")
    print(f"Input:  {input_dir}")
    print(f"Output: {train_dir}")
    print(f"SFT size: {sft_size:,}")
    print(f"Test ratio: {test_ratio}")
    if dry_run:
        print("DRY RUN — no files will be written")

    # Load all generators
    generator_data = load_all_generators(input_dir)
    if not generator_data:
        print("No generator data found!")
        return {}

    print(f"\nLoaded {len(generator_data)} generator files:")
    for name, convs in sorted(generator_data.items()):
        print(f"  {name}: {len(convs):,} conversations")

    # Separate train-only generators (e.g. gen_h_histfacts — factual recall
    # is evaluated via external benchmarks, not a held-out test split)
    TRAIN_ONLY_PREFIXES = ("gen_h_histfacts",)
    splittable_data = {}
    train_only_data = {}
    for name, convs in generator_data.items():
        if any(name.startswith(p) for p in TRAIN_ONLY_PREFIXES):
            train_only_data[name] = convs
            print(f"  (train-only: {name})")
        else:
            splittable_data[name] = convs

    train_only_convs = []
    for convs in train_only_data.values():
        train_only_convs.extend(convs)

    # Merge splittable generators
    all_splittable = []
    for convs in splittable_data.values():
        all_splittable.extend(convs)

    total_splittable = len(all_splittable)
    print(f"\nSplittable: {total_splittable:,} conversations")
    print(f"Train-only: {len(train_only_convs):,} conversations (gen_h)")

    # Train/test split on splittable data
    rng = random.Random(SEED)
    indices = list(range(total_splittable))
    rng.shuffle(indices)
    test_size = max(1, int(total_splittable * test_ratio))
    test_indices = set(indices[:test_size])

    test = [all_splittable[i] for i in range(total_splittable) if i in test_indices]
    train_splittable = [all_splittable[i] for i in range(total_splittable) if i not in test_indices]

    # Mid-train = splittable train + all train-only
    midtrain = train_splittable + train_only_convs
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(midtrain)

    # SFT = proportional subsample from TRAIN data only (excluding test + train-only)
    # Must sample from post-split train data to avoid SFT/test contamination
    test_ids = set(id(c) for c in test)
    train_by_gen = {}
    for name, convs in splittable_data.items():
        train_convs = [c for c in convs if id(c) not in test_ids]
        if train_convs:
            train_by_gen[name] = train_convs

    sft = proportional_subsample(
        train_by_gen, sft_size,
        seed=SEED + 2,
    )

    total = len(midtrain) + len(test)
    print(f"\nTotal: {total:,}")
    print(f"  Mid-train: {len(midtrain):,}")
    print(f"  SFT:       {len(sft):,}")
    print(f"  Test:      {len(test):,}")

    # Write output files
    if not dry_run:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        midtrain_path = train_dir / "hist_synthetic_midtrain.jsonl"
        sft_path = train_dir / "hist_synthetic_sft.jsonl"
        test_path = test_dir / "hist_synthetic_test.jsonl"

        write_jsonl(midtrain, str(midtrain_path), validate=False)
        write_jsonl(sft, str(sft_path), validate=False)
        write_jsonl(test, str(test_path), validate=False)

        print(f"\nWritten:")
        print(f"  {midtrain_path}")
        print(f"  {sft_path}")
        print(f"  {test_path}")

    stats = {
        "input_dir": str(input_dir),
        "generators": {name: len(convs) for name, convs in generator_data.items()},
        "total": total,
        "midtrain": len(midtrain),
        "sft": len(sft),
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
    print(f"{'Mid-train':<30} {len(midtrain):>10,}")
    print(f"{'SFT':<30} {len(sft):>10,}")
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
    parser.add_argument("--sft-size", type=int, default=DEFAULT_SFT_SIZE,
                        help=f"SFT subsample size (default: {DEFAULT_SFT_SIZE:,})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without writing files")
    args = parser.parse_args()

    assemble(args.period, source=args.source, test_ratio=args.test_ratio,
             sft_size=args.sft_size, dry_run=args.dry_run)
