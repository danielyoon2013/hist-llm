"""
Quality pipeline orchestrator: validate -> dedup -> (LAB filter via filter.py) -> split.

Each step reads from the previous step's output directory.
"""

import os
import json
from pathlib import Path

from src.post_training.config import get_paths
from src.post_training.quality.validate import validate_directory
from src.post_training.quality.dedup import dedup_directory


def run_pipeline(period, steps=None):
    """Run the quality pipeline for a period.

    Args:
        period: Period key (e.g. "1900_1949")
        steps: List of steps to run. None = all.
               Options: ["validate", "dedup"]
               LAB filter and split are run via their own CLIs.

    Returns:
        dict of pipeline stats
    """
    paths = get_paths(period)
    generators_dir = paths["synthetic_dir"] / "by_generator"
    quality_dir = paths["synthetic_dir"].parent / "quality"
    validated_dir = quality_dir / "validated"
    deduped_dir = quality_dir / "deduped"
    stats_path = quality_dir / "stats.json"

    os.makedirs(quality_dir, exist_ok=True)

    if steps is None:
        steps = ["validate", "dedup"]

    all_stats = {}

    # Step 1: Validate
    if "validate" in steps:
        print(f"\n{'='*70}")
        print(f"Step 1: Validate")
        print(f"  Input:  {generators_dir}")
        print(f"  Output: {validated_dir}")
        print(f"{'='*70}")

        if not generators_dir.exists():
            print(f"  ERROR: {generators_dir} does not exist. Run generate.py first.")
            return all_stats

        val_stats = validate_directory(generators_dir, validated_dir)
        all_stats["validate"] = val_stats

    # Step 2: Dedup
    if "dedup" in steps:
        print(f"\n{'='*70}")
        print(f"Step 2: Deduplicate")

        # Use validated output if available, else raw generator output
        dedup_input = validated_dir if validated_dir.exists() else generators_dir
        print(f"  Input:  {dedup_input}")
        print(f"  Output: {deduped_dir}")
        print(f"{'='*70}")

        dedup_stats = dedup_directory(dedup_input, deduped_dir)
        all_stats["dedup"] = dedup_stats

    # Save stats
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    # Summary
    print(f"\n{'='*70}")
    print("Pipeline Summary")
    print(f"{'='*70}")
    if "validate" in all_stats:
        total_in = sum(s["total"] for s in all_stats["validate"].values())
        total_valid = sum(s["valid"] for s in all_stats["validate"].values())
        print(f"  Validate: {total_in} -> {total_valid} "
              f"({100*total_valid/total_in:.1f}% pass)" if total_in else "  Validate: 0 inputs")
    if "dedup" in all_stats:
        total_orig = sum(s.get("original", 0) for s in all_stats["dedup"].values())
        total_final = sum(s.get("final", 0) for s in all_stats["dedup"].values())
        print(f"  Dedup: {total_orig} -> {total_final} "
              f"({100*total_final/total_orig:.1f}% kept)" if total_orig else "  Dedup: 0 inputs")

    return all_stats
