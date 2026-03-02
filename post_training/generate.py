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
    python -m src.post_training.generate --period 1900_1949 --target 120 --sync

    # Run specific generators only
    python -m src.post_training.generate submit --period 1900_1949 --generators A B D

Allocation is derived from format counts (equal weight per format slot).
At --target 1,000,000 with 9 format slots (111,111 per slot):
    A (Factual QA):       222,222  — corpus
    B (Chain-of-Thought): 222,222  — corpus
    C (Comprehension):    111,111  — corpus
    D (Quantitative):     222,222  — corpus
    E (Completion):       111,111  — corpus
    F (Instruct):         111,111  — corpus
"""

import argparse

from src.post_training.config import (
    PERIODS, DEFAULT_TARGET, GENERATOR_SPEC, compute_plan, get_paths,
)
from src.post_training.generators import get_generator_registry


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
                        choices=list(GENERATOR_SPEC.keys()),
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

    action = "run" if args.sync else args.action
    gen_keys = args.generators or list(GENERATOR_SPEC.keys())

    # Single call derives everything
    plan = compute_plan(args.target, gen_keys)

    # Print header
    action_label = {"submit": "SUBMIT (Batch API)", "check": "CHECK STATUS",
                    "process": "PROCESS RESULTS", "run": "SYNC MODE"}
    print(f"{'='*70}")
    print(f"Synthetic Data Generation — {action_label[action]}")
    print(f"Period: {args.period} ({PERIODS[args.period][0]}-{PERIODS[args.period][1]})")
    print(f"Target: {args.target:,} total examples")
    print(f"{'='*70}")
    print(f"\n{'Generator':<12} {'Target':>10} {'Docs':>10}")
    print(f"{'-'*40}")
    for g in sorted(gen_keys):
        docs = plan["generators"][g]["docs_needed"]
        docs_str = f"{docs:,}" if docs else "—"
        print(f"  {g:<10} {plan['generators'][g]['target']:>10,} {docs_str:>10}")
    print(f"{'-'*40}")
    total = sum(plan["generators"][g]["target"] for g in gen_keys)
    print(f"  {'TOTAL':<10} {total:>10,}")
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
        gen_plan = plan["generators"][gen_key]

        # Per-generator docs_needed (None for metadata generators)
        docs_for_gen = gen_plan["docs_needed"]
        if args.max_docs is not None:
            docs_for_gen = args.max_docs  # legacy override

        print(f"\n{'='*70}")
        verb = {"submit": "Submitting", "process": "Processing", "run": "Running"}
        print(f"{verb[action]} Generator {gen_key}: {gen.name}")
        print(f"Target: {gen_plan['target']:,} examples")
        print(f"{'='*70}")

        result = gen.run(
            period=args.period,
            collections=args.collections,
            max_workers=args.max_workers,
            max_docs=docs_for_gen,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            target_examples=gen_plan["target"],
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
