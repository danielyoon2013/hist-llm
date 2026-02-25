"""
Main entry point for the quality pipeline.

Usage:
    # Run full pipeline (validate + dedup)
    python -m src.post_training.process --period 1900_1949

    # Run specific steps
    python -m src.post_training.process --period 1900_1949 --step validate
    python -m src.post_training.process --period 1900_1949 --step dedup

    # LAB filtering is done separately via filter.py:
    python -m src.post_training.instruct.filter --period 1900_1949 --submit --corpus
    python -m src.post_training.instruct.filter --period 1900_1949 --process --corpus
"""

import argparse

from src.post_training.config import PERIODS
from src.post_training.quality.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Quality pipeline: validate and deduplicate synthetic data"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--step", type=str, nargs="+", default=None,
                        choices=["validate", "dedup"],
                        help="Specific steps to run (default: all)")
    args = parser.parse_args()

    run_pipeline(args.period, steps=args.step)


if __name__ == "__main__":
    main()
