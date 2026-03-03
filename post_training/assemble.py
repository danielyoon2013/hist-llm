"""
Assemble synthetic data into nanochat-ready training files.

Reads per-generator JSONL from quality/deduped/ (or by_generator/),
performs a DOCUMENT-LEVEL train/test split, and produces:
  1. hist_synthetic_midtrain.jsonl  — all training examples (all formats)
  2. hist_synthetic_sft.jsonl       — proportional subsample (10K target)
  3. hist_synthetic_test.jsonl      — MC-only holdout from held-out documents

Document-level split ensures no content leakage: test questions come from
documents never seen during training (in any format).

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

from src.post_training.config import (
    PERIODS, get_paths, DEFAULT_SFT_SIZE, DEFAULT_TEST_RATIO,
    DEFAULT_TARGET, GENERATOR_SPEC,
)
from src.post_training.utils import read_jsonl, write_jsonl


SEED = 42

# MC format keys — only these go into the test split
MC_FORMATS = {"mc4", "mc4_passage"}


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
    Each conversation may be a bare list (legacy) or dict with metadata.
    """
    data = {}
    for jsonl_file in sorted(Path(input_dir).glob("*.jsonl")):
        name = jsonl_file.stem
        conversations = read_jsonl(str(jsonl_file))
        if conversations:
            data[name] = conversations
    return data


def _extract_messages(conv):
    """Extract bare message list from a conversation (handles both formats)."""
    if isinstance(conv, dict):
        return conv["messages"]
    return conv


def _extract_doc_name(conv):
    """Extract doc_name from a conversation (returns 'unknown' for legacy format)."""
    if isinstance(conv, dict):
        return conv.get("doc_name", "unknown")
    return "unknown"


def _extract_format(conv):
    """Extract format key from a conversation."""
    if isinstance(conv, dict):
        return conv.get("format", "")
    return ""


def proportional_subsample(generator_data, target_size, seed=SEED):
    """Subsample proportionally from each generator to hit target_size.

    Args:
        generator_data: dict of {gen_name: [messages_lists]}
        target_size: total examples to sample
        seed: random seed

    Returns:
        list of sampled message lists (shuffled)
    """
    rng = random.Random(seed)

    total_eligible = sum(len(c) for c in generator_data.values())
    if total_eligible == 0:
        return []

    samples = []
    remaining = target_size
    gen_items = sorted(generator_data.items())

    for i, (name, convs) in enumerate(gen_items):
        if i == len(gen_items) - 1:
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


def downsample_to_target(generator_data, target, seed=SEED):
    """Downsample each format-slot file to equal share of target.

    Each JSONL file corresponds to one format slot. All slots get
    target // total_slots conversations. Files under the cap pass through.

    Args:
        generator_data: dict of {filename_stem: [conversations]}
        target: total conversation target (e.g. 1_000_000)
        seed: random seed for reproducible sampling

    Returns:
        generator_data (modified in place), with before/after counts printed.
    """
    total_slots = sum(len(spec["formats"]) for spec in GENERATOR_SPEC.values())
    per_slot = target // total_slots

    rng = random.Random(seed)
    total_before = 0
    total_after = 0

    print(f"\nDownsampling to {target:,} total ({per_slot:,} per format slot, {total_slots} slots):")
    for name in sorted(generator_data.keys()):
        before = len(generator_data[name])
        total_before += before
        if before > per_slot:
            generator_data[name] = rng.sample(generator_data[name], per_slot)
        after = len(generator_data[name])
        total_after += after
        status = f"{before:,} -> {after:,}" if before > per_slot else f"{before:,} (kept)"
        print(f"  {name}: {status}")

    print(f"  Total: {total_before:,} -> {total_after:,}")
    return generator_data


def assemble(period, source=None, test_ratio=DEFAULT_TEST_RATIO,
             sft_size=DEFAULT_SFT_SIZE, target=DEFAULT_TARGET, dry_run=False):
    """Assemble synthetic data for a period with document-level train/test split.

    Split strategy:
      - 95% of unique documents → TRAIN (all formats: mc, open, cot, passage)
      - 5% of unique documents → TEST (MC formats only, with letters for eval)

    This ensures:
      - No content leakage between train and test
      - Format diversity in training data
      - MC-only test set for fast categorical evaluation

    Produces three output files in final/:
      - train/hist_synthetic_midtrain.jsonl  (all train examples, all formats)
      - train/hist_synthetic_sft.jsonl       (proportional subsample for SFT)
      - test/hist_synthetic_test.jsonl       (MC-only from held-out documents)
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
    print(f"Output: train={train_dir}, test={test_dir}")
    print(f"Target: {target:,}")
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
    total_all = 0
    for name, convs in sorted(generator_data.items()):
        print(f"  {name}: {len(convs):,} conversations")
        total_all += len(convs)

    # ------------------------------------------------------------------
    # Downsample to target (cap each format slot equally)
    # ------------------------------------------------------------------
    generator_data = downsample_to_target(generator_data, target)

    # ------------------------------------------------------------------
    # Document-level split
    # ------------------------------------------------------------------

    # Collect unique doc_names across all generators
    doc_names = set()
    for convs in generator_data.values():
        for conv in convs:
            doc_names.add(_extract_doc_name(conv))

    doc_list = sorted(doc_names)
    rng = random.Random(SEED)
    rng.shuffle(doc_list)

    test_doc_count = max(1, int(len(doc_list) * test_ratio))
    test_docs = set(doc_list[:test_doc_count])
    train_docs = set(doc_list[test_doc_count:])

    print(f"\nDocuments: {len(doc_list)} unique")
    print(f"  Train docs: {len(train_docs)}")
    print(f"  Test docs:  {len(test_docs)}")

    # ------------------------------------------------------------------
    # Split conversations by document assignment
    # ------------------------------------------------------------------

    train_by_gen = defaultdict(list)   # gen_filename -> [bare_messages]
    test_convs = []                    # [{messages, letters}] for eval

    train_count = 0
    test_count = 0
    test_skipped_non_mc = 0

    for gen_name, convs in generator_data.items():
        for conv in convs:
            doc = _extract_doc_name(conv)
            fmt = _extract_format(conv)
            messages = _extract_messages(conv)

            if doc in test_docs:
                if fmt in MC_FORMATS:
                    test_convs.append({
                        "messages": messages,
                        "letters": ["A", "B", "C", "D"],
                    })
                    test_count += 1
                else:
                    test_skipped_non_mc += 1
            else:
                # Train: bare message lists for nanochat
                train_by_gen[gen_name].append(messages)
                train_count += 1

    # ------------------------------------------------------------------
    # Build output sets
    # ------------------------------------------------------------------

    # Midtrain = all train convs shuffled
    midtrain = []
    for convs in train_by_gen.values():
        midtrain.extend(convs)
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(midtrain)

    # SFT = proportional subsample from train
    sft = proportional_subsample(train_by_gen, sft_size, seed=SEED + 2)

    # Shuffle test
    rng3 = random.Random(SEED + 3)
    rng3.shuffle(test_convs)

    print(f"\nSplit results:")
    print(f"  Mid-train: {len(midtrain):,} (all formats from train docs)")
    print(f"  SFT:       {len(sft):,} (proportional subsample)")
    print(f"  Test:      {len(test_convs):,} (MC-only from test docs)")
    print(f"  Discarded: {test_skipped_non_mc:,} (non-MC from test docs)")

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------

    if not dry_run:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        midtrain_path = train_dir / "hist_synthetic_midtrain.jsonl"
        sft_path = train_dir / "hist_synthetic_sft.jsonl"
        test_path = test_dir / "hist_synthetic_test.jsonl"

        write_jsonl(midtrain, str(midtrain_path), validate=False)
        write_jsonl(sft, str(sft_path), validate=False)
        write_jsonl(test_convs, str(test_path), validate=False)

        print(f"\nWritten:")
        print(f"  {midtrain_path}")
        print(f"  {sft_path}")
        print(f"  {test_path}")

    stats = {
        "input_dir": str(input_dir),
        "generators": {name: len(convs) for name, convs in generator_data.items()},
        "documents": len(doc_list),
        "train_docs": len(train_docs),
        "test_docs": len(test_docs),
        "midtrain": len(midtrain),
        "sft": len(sft),
        "test": len(test_convs),
        "test_skipped_non_mc": test_skipped_non_mc,
    }

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Generator':<35} {'Train':>8} {'Test':>8}")
    print(f"{'-'*60}")
    gen_test_counts = defaultdict(int)
    for conv in test_convs:
        # Count by format for test
        gen_test_counts["test_mc"] += 1

    for name in sorted(generator_data.keys()):
        train_n = len(train_by_gen.get(name, []))
        print(f"{name:<35} {train_n:>8,}")
    print(f"{'-'*60}")
    print(f"{'TRAIN TOTAL':<35} {len(midtrain):>8,}")
    print(f"{'SFT (subsample)':<35} {len(sft):>8,}")
    print(f"{'TEST (MC from held-out docs)':<35} {'':>8} {len(test_convs):>8,}")

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
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Total conversation target (default: {DEFAULT_TARGET:,})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without writing files")
    args = parser.parse_args()

    assemble(args.period, source=args.source, test_ratio=args.test_ratio,
             sft_size=args.sft_size, target=args.target, dry_run=args.dry_run)
