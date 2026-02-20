"""
Score instruct datasets for temporal contamination via OpenAI live API.

For each row in each dataset, asks GPT-4o-mini whether it requires knowledge
that didn't exist before the period's end year. Saves per-row scores to a
separate file so they can be analyzed and used for filtering independently.

Usage:
    python -m src.post_training.score_instruct --period 1950_1999
    python -m src.post_training.score_instruct --period 1950_1999 --dataset gsm8k
    python -m src.post_training.score_instruct --period 1950_1999 --dry-run
    python -m src.post_training.score_instruct --period 1950_1999 --workers 32
"""

import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.post_training.config import PERIODS, get_paths, PROJECT_ROOT
from src.post_training.utils import call_openai_json, count_jsonl
from src.post_training.instruct.filter import (
    extract_text_for_classification,
    build_classification_prompt,
)


INSTRUCT_DIR = PROJECT_ROOT / "data" / "instruct_data"
DEFAULT_WORKERS = 16
CHUNK_SIZE = 1000  # rows per processing chunk (for ordered writes + resume)

ALL_DATASETS = [
    "smoltalk", "mmlu", "arc_easy", "arc_challenge", "gsm8k",
    "math", "aime_amc", "commonsenseqa", "hellaswag", "piqa",
    "winogrande", "logiqa", "folio", "hotpotqa", "musique",
    "strategyqa", "scienceqa", "humaneval", "mbpp", "codecontests",
]


def score_one_row(index, messages, end_year):
    """Classify one conversation row via live API."""
    text = extract_text_for_classification(messages)
    prompt = build_classification_prompt(text, end_year)
    result = call_openai_json(
        [{"role": "user", "content": prompt}],
        max_tokens=64,
    )
    keep = result.get("keep", True)  # default to keep on parse weirdness
    return {"index": index, "keep": keep}


def score_dataset(dataset_name, end_year, LAB_scores_dir, workers, dry_run=False):
    """Score all rows in a dataset, saving to a score file. Supports resuming."""
    input_path = INSTRUCT_DIR / f"{dataset_name}.jsonl"
    score_path = LAB_scores_dir / f"{dataset_name}_scores.jsonl"

    if not input_path.exists():
        print(f"  {dataset_name}: not found at {input_path}, skipping")
        return

    # Read all conversations
    conversations = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    total = len(conversations)

    # Check for existing scores (resume support)
    existing = 0
    if score_path.exists():
        existing = count_jsonl(str(score_path))
    if existing >= total:
        print(f"  {dataset_name}: already complete ({existing}/{total})")
        return
    if existing > 0:
        print(f"  {dataset_name}: resuming from row {existing}/{total}")

    if dry_run:
        sample = conversations[existing:existing + 5]
        for i, msgs in enumerate(sample):
            idx = existing + i
            result = score_one_row(idx, msgs, end_year)
            status = "KEEP" if result["keep"] else "REMOVE"
            preview = msgs[0]["content"][:80] if msgs else ""
            print(f"  [{status}] row {idx}: {preview}...")
        return

    # Score remaining rows in chunks for ordered writes
    remaining = conversations[existing:]
    scored = 0
    errors = 0
    start_time = time.time()

    os.makedirs(LAB_scores_dir, exist_ok=True)

    for chunk_start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_results = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    score_one_row,
                    existing + chunk_start + i,
                    msgs,
                    end_year,
                ): existing + chunk_start + i
                for i, msgs in enumerate(chunk)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    chunk_results.append(result)
                    scored += 1
                except Exception as e:
                    chunk_results.append({"index": idx, "keep": True})
                    errors += 1
                    print(f"  Error on row {idx}: {e}")

        # Sort by index then append to file
        chunk_results.sort(key=lambda x: x["index"])
        with open(score_path, "a", encoding="utf-8") as f:
            for item in chunk_results:
                f.write(json.dumps(item) + "\n")

        # Progress reporting
        total_done = existing + scored
        elapsed = time.time() - start_time
        rate = scored / elapsed if elapsed > 0 else 0
        eta = (total - total_done) / rate if rate > 0 else 0
        print(f"  {dataset_name}: {total_done:,}/{total:,} "
              f"({rate:.1f} rows/s, ETA {eta/60:.0f}m, {errors} errors)")

    final_count = count_jsonl(str(score_path))
    print(f"  {dataset_name}: done ({final_count:,}/{total:,}, {errors} errors)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score instruct datasets for temporal contamination via live API"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--dataset", type=str, default=None,
                        choices=ALL_DATASETS,
                        help="Score a single dataset (default: all)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel API workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score 5 examples per dataset and print results")
    args = parser.parse_args()

    paths = get_paths(args.period)
    end_year = paths["end_year"]
    LAB_scores_dir = paths["LAB_scores_dir"]

    print(f"Period: {args.period} ({paths['start_year']}-{end_year})")
    print(f"Scores dir: {LAB_scores_dir}")
    print(f"Workers: {args.workers}")

    datasets = [args.dataset] if args.dataset else ALL_DATASETS

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Scoring: {dataset_name}")
        print(f"{'='*60}")
        score_dataset(dataset_name, end_year, LAB_scores_dir, args.workers,
                      args.dry_run)

    if not args.dry_run:
        print(f"\nDone! Score files saved to: {LAB_scores_dir}")
        print("Run analyze_scores.py --stats to see contamination rates.")
