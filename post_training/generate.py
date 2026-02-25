"""
Main entry point for running synthetic data generators.

Usage:
    # Run all generators for a period (tiny test)
    python -m src.post_training.generate --period 1900_1949 --generators A B C D E F G H --max-docs 3

    # Run specific generators
    python -m src.post_training.generate --period 1900_1949 --generators A B --max-docs 3

    # Metadata-based generators (D, H) don't need --max-docs
    python -m src.post_training.generate --period 1900_1949 --generators D H
"""

import argparse

from src.post_training.config import PERIODS
from src.post_training.generators import get_generator_registry


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic data generators for historical LLM post-training"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--generators", type=str, nargs="+", required=True,
                        choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                        help="Which generators to run (A-H)")
    parser.add_argument("--collections", type=str, nargs="+", default=None,
                        help="Specific collections to process (default: all)")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Concurrent API calls (default: 50)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit documents per collection (for testing)")
    parser.add_argument("--chunk-size", type=int, default=6000,
                        help="Characters per chunk (default: 6000)")
    parser.add_argument("--overlap", type=int, default=300,
                        help="Overlap between chunks (default: 300)")
    args = parser.parse_args()

    registry = get_generator_registry()

    for gen_key in args.generators:
        gen_cls = registry[gen_key]
        gen = gen_cls()
        print(f"\n{'='*70}")
        print(f"Running Generator {gen_key}: {gen.name}")
        print(f"{'='*70}")

        output_path = gen.run(
            period=args.period,
            collections=args.collections,
            max_workers=args.max_workers,
            max_docs=args.max_docs,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

        if output_path:
            print(f"Output: {output_path}")
        else:
            print(f"Generator {gen_key} produced no output.")

    print(f"\n{'='*70}")
    print("All generators complete.")


if __name__ == "__main__":
    main()
