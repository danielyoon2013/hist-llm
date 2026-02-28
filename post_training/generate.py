"""
Main entry point for running synthetic data generators.

Simplified pipeline: one command generates all data for a period.

Usage:
    # Full run: 1M examples for a period (default)
    python -m src.post_training.generate --period 1900_1949

    # Custom target
    python -m src.post_training.generate --period 1900_1949 --target 500000

    # Tiny test run (auto-computes ~3 docs worth)
    python -m src.post_training.generate --period 1900_1949 --target 180

    # Run specific generators only
    python -m src.post_training.generate --period 1900_1949 --generators A B D H

    # Legacy: explicit doc count (overrides auto-computation)
    python -m src.post_training.generate --period 1900_1949 --max-docs 3

Allocation at --target 1,000,000 (approximate, ±1 from rounding):
    A (Factual QA):      ~190,000  (19.0%)  — corpus
    B (Chain-of-Thought): 190,000  (19.0%)  — corpus
    C (Comprehension):   190,000  (19.0%)  — corpus
    E (Quantitative):    ~126,666  (12.7%)  — corpus
    F (Completion):      190,000  (19.0%)  — corpus
    G (Instruct):         63,333   (6.3%)  — corpus
    D (Temporal):         25,000   (2.5%)  — metadata
    H (Hist Facts):       25,000   (2.5%)  — metadata, train-only
"""

import argparse

from src.post_training.config import (
    PERIODS, DEFAULT_TARGET, EXAMPLES_PER_DOC, compute_allocation,
)
from src.post_training.generators import get_generator_registry


CORPUS_GENERATORS = {"A", "B", "C", "E", "F", "G"}
METADATA_GENERATORS = {"D", "H"}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for a historical period"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Total examples to generate (default: {DEFAULT_TARGET:,})")
    parser.add_argument("--generators", type=str, nargs="+", default=None,
                        choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                        help="Which generators to run (default: all)")
    parser.add_argument("--collections", type=str, nargs="+", default=None,
                        help="Specific collections to process (default: all)")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Concurrent API calls (default: 50)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Override auto-computed doc count (legacy)")
    parser.add_argument("--chunk-size", type=int, default=6000,
                        help="Characters per chunk (default: 6000)")
    parser.add_argument("--overlap", type=int, default=300,
                        help="Overlap between chunks (default: 300)")
    args = parser.parse_args()

    gen_keys = args.generators or ["A", "B", "C", "D", "E", "F", "G", "H"]

    # Compute per-generator allocation
    allocation = compute_allocation(args.target)

    # Compute docs needed for corpus generators
    corpus_target = sum(allocation[g] for g in CORPUS_GENERATORS if g in gen_keys)
    docs_needed = max(1, -(-corpus_target // EXAMPLES_PER_DOC))  # ceil div

    if args.max_docs is not None:
        docs_needed = args.max_docs  # legacy override

    # Print allocation summary
    print(f"{'='*70}")
    print(f"Synthetic Data Generation")
    print(f"Period: {args.period} ({PERIODS[args.period][0]}-{PERIODS[args.period][1]})")
    print(f"Target: {args.target:,} total examples")
    print(f"Docs needed: {docs_needed:,} (at {EXAMPLES_PER_DOC} examples/doc)")
    print(f"{'='*70}")
    print(f"\n{'Generator':<25} {'Type':<10} {'Target':>10}")
    print(f"{'-'*45}")
    for g in sorted(gen_keys):
        gtype = "metadata" if g in METADATA_GENERATORS else "corpus"
        print(f"  {g:<23} {gtype:<10} {allocation[g]:>10,}")
    print(f"{'-'*45}")
    print(f"  {'TOTAL':<23} {'':10} {sum(allocation[g] for g in gen_keys):>10,}")
    print()

    registry = get_generator_registry()
    total_files = 0
    total_conversations = 0

    for gen_key in gen_keys:
        gen_cls = registry[gen_key]
        gen = gen_cls()
        target_for_gen = allocation[gen_key]

        print(f"\n{'='*70}")
        print(f"Running Generator {gen_key}: {gen.name}")
        print(f"Target: {target_for_gen:,} examples")
        print(f"Formats: {list(gen.SUPPORTED_FORMATS)}")
        print(f"{'='*70}")

        output_paths = gen.run(
            period=args.period,
            collections=args.collections,
            max_workers=args.max_workers,
            max_docs=docs_needed if gen.needs_corpus else None,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            target_examples=target_for_gen,
        )

        if output_paths:
            for fmt, path in output_paths.items():
                total_files += 1
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        count = sum(1 for _ in f)
                    total_conversations += count
                except Exception:
                    pass
        else:
            print(f"Generator {gen_key} produced no output.")

    print(f"\n{'='*70}")
    print(f"All generators complete.")
    print(f"Target: {args.target:,} | Actual: {total_conversations:,} "
          f"({total_files} output files)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
