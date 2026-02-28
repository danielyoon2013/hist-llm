"""
Main entry point for running synthetic data generators.

Supports two execution modes:
- Batch API (submit/check/process): 50% cost savings, ~24h turnaround
- Sync API (--sync or default "run"): ThreadPoolExecutor, instant results

Usage:
    # Batch mode (production) — 3-step workflow
    python -m src.post_training.generate submit --period 1900_1949
    python -m src.post_training.generate check --period 1900_1949
    python -m src.post_training.generate process --period 1900_1949

    # Sync mode (testing / small runs)
    python -m src.post_training.generate --period 1900_1949 --target 180 --sync

    # Sync mode is also the default action
    python -m src.post_training.generate --period 1900_1949 --target 180

    # Run specific generators only
    python -m src.post_training.generate submit --period 1900_1949 --generators A B D H

    # Legacy: explicit doc count (overrides auto-computation)
    python -m src.post_training.generate --period 1900_1949 --max-docs 3 --sync

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
    PERIODS, DEFAULT_TARGET, EXAMPLES_PER_DOC, compute_allocation, get_paths,
)
from src.post_training.generators import get_generator_registry


CORPUS_GENERATORS = {"A", "B", "C", "E", "F", "G"}
METADATA_GENERATORS = {"D", "H"}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for a historical period"
    )
    parser.add_argument("action", nargs="?", default="run",
                        choices=["submit", "check", "process", "run"],
                        help="Batch workflow step (default: run = sync mode)")
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
                        help="Concurrent API calls for sync mode (default: 50)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Override auto-computed doc count (legacy)")
    parser.add_argument("--chunk-size", type=int, default=6000,
                        help="Characters per chunk (default: 6000)")
    parser.add_argument("--overlap", type=int, default=300,
                        help="Overlap between chunks (default: 300)")
    parser.add_argument("--sync", action="store_true",
                        help="Force synchronous API calls (for small test runs)")
    args = parser.parse_args()

    # --sync forces sync mode regardless of action
    action = "run" if args.sync else args.action

    gen_keys = args.generators or ["A", "B", "C", "D", "E", "F", "G", "H"]

    # Compute per-generator allocation
    allocation = compute_allocation(args.target)

    # Compute docs needed for corpus generators
    corpus_target = sum(allocation[g] for g in CORPUS_GENERATORS if g in gen_keys)
    docs_needed = max(1, -(-corpus_target // EXAMPLES_PER_DOC))  # ceil div

    if args.max_docs is not None:
        docs_needed = args.max_docs  # legacy override

    # Print header
    action_label = {"submit": "SUBMIT (Batch API)", "check": "CHECK STATUS",
                    "process": "PROCESS RESULTS", "run": "SYNC MODE"}
    print(f"{'='*70}")
    print(f"Synthetic Data Generation — {action_label[action]}")
    print(f"Period: {args.period} ({PERIODS[args.period][0]}-{PERIODS[args.period][1]})")
    print(f"Target: {args.target:,} total examples")
    if action in ("submit", "run"):
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

    # --- CHECK action: just print batch statuses ---
    if action == "check":
        from src.post_training.utils import check_batch_status
        paths = get_paths(args.period)
        batch_dir = paths["batch_temp_dir"]
        for gen_key in gen_keys:
            gen_cls = registry[gen_key]
            gen = gen_cls()
            id_path = batch_dir / f"{gen.name}_batch_id.txt"
            if id_path.exists():
                batch_id = id_path.read_text().strip()
                print(f"\n{gen.name}:")
                check_batch_status(batch_id)
            else:
                print(f"\n{gen.name}: no batch submitted")
        return

    # --- SUBMIT / PROCESS / RUN actions ---
    total_files = 0
    total_conversations = 0

    for gen_key in gen_keys:
        gen_cls = registry[gen_key]
        gen = gen_cls()
        target_for_gen = allocation[gen_key]

        print(f"\n{'='*70}")
        verb = {"submit": "Submitting", "process": "Processing", "run": "Running"}
        print(f"{verb[action]} Generator {gen_key}: {gen.name}")
        print(f"Target: {target_for_gen:,} examples")
        print(f"{'='*70}")

        result = gen.run(
            period=args.period,
            collections=args.collections,
            max_workers=args.max_workers,
            max_docs=docs_needed if gen.needs_corpus else None,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            target_examples=target_for_gen,
            action=action,
        )

        if action == "submit":
            if result:
                print(f"  Batch submitted: {result}")
            else:
                print(f"  Generator {gen_key}: submit failed.")
        elif result and isinstance(result, dict):
            for fmt, path in result.items():
                total_files += 1
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        count = sum(1 for _ in f)
                    total_conversations += count
                except Exception:
                    pass
        else:
            print(f"Generator {gen_key} produced no output.")

    if action in ("process", "run"):
        print(f"\n{'='*70}")
        print(f"All generators complete.")
        print(f"Target: {args.target:,} | Actual: {total_conversations:,} "
              f"({total_files} output files)")
        print(f"{'='*70}")
    elif action == "submit":
        print(f"\n{'='*70}")
        print(f"All batches submitted. Run 'check' to monitor status.")
        print(f"When complete, run 'process' to download and write output files.")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
