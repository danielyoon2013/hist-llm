"""
Analyze temporal contamination scores and filter datasets.

Three modes:
  --import-batch  Import scores from existing OpenAI Batch API results
  --stats         Print per-dataset contamination statistics
  --filter        Create filtered JSONL files based on scores

Usage:
    python -m src.post_training.analyze_scores --period 1950_1999 --import-batch
    python -m src.post_training.analyze_scores --period 1950_1999 --stats
    python -m src.post_training.analyze_scores --period 1950_1999 --filter
"""

import os
import json
import argparse

from src.post_training.config import PERIODS, get_paths, DATA_ROOT
from src.post_training.utils import read_jsonl, write_jsonl, download_batch_results
from src.post_training.instruct.score import ALL_DATASETS


INSTRUCT_DIR = DATA_ROOT / "instruct_data"


# ---------------------------------------------------------------------------
# Import batch results into standard score format
# ---------------------------------------------------------------------------

def import_batch_scores(period):
    """
    Convert existing Batch API results into the standard score file format.
    Reads batch results from batch_temp/, writes to scores/{dataset}_scores.jsonl.
    """
    paths = get_paths(period)
    batch_dir = paths["posttraining_dir"] / "batch_temp"
    LAB_scores_dir = paths["LAB_scores_dir"]
    os.makedirs(LAB_scores_dir, exist_ok=True)

    BATCH_DATASETS = ["smoltalk", "mmlu", "arc_easy", "arc_challenge"]

    for dataset_name in BATCH_DATASETS:
        id_file = batch_dir / f"{dataset_name}_batch_ids.txt"
        if not id_file.exists():
            print(f"  {dataset_name}: no batch IDs found, skipping")
            continue

        batch_ids = [line.strip() for line in open(id_file) if line.strip()]

        # Download all chunk results
        all_results = []
        all_complete = True
        for i, batch_id in enumerate(batch_ids):
            suffix = f"_chunk{i}" if len(batch_ids) > 1 else ""
            results_file = batch_dir / f"{dataset_name}{suffix}_results.jsonl"

            # If already downloaded, read from file
            if results_file.exists():
                results = []
                with open(results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line.strip())
                        custom_id = data["custom_id"]
                        body = data["response"]["body"]["choices"][0]["message"]["content"]
                        results.append((custom_id, body))
            else:
                results = download_batch_results(batch_id, str(results_file))
                if results is None:
                    print(f"  {dataset_name}: chunk {i+1}/{len(batch_ids)} not complete yet")
                    all_complete = False
                    break
            all_results.extend(results)

        if not all_complete:
            continue

        # Count original rows
        input_path = INSTRUCT_DIR / f"{dataset_name}.jsonl"
        total_rows = sum(1 for line in open(input_path, "r", encoding="utf-8")
                         if line.strip())

        # Build index -> keep mapping from batch results
        # custom_id format: "{dataset_name}_{index}" e.g. "arc_easy_0"
        decisions = {}
        for custom_id, response_text in all_results:
            idx = int(custom_id.rsplit("_", 1)[1])
            try:
                parsed = json.loads(response_text)
                decisions[idx] = parsed.get("keep", True)
            except json.JSONDecodeError:
                decisions[idx] = True  # conservative default

        # Write score file (one line per row, in order)
        score_path = LAB_scores_dir / f"{dataset_name}_scores.jsonl"
        with open(score_path, "w", encoding="utf-8") as f:
            for i in range(total_rows):
                keep = decisions.get(i, True)
                f.write(json.dumps({"index": i, "keep": keep}) + "\n")

        matched = sum(1 for i in range(total_rows) if i in decisions)
        removed = sum(1 for v in decisions.values() if not v)
        print(f"  {dataset_name}: imported {matched:,}/{total_rows:,} scores "
              f"({removed:,} flagged for removal)")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(period):
    """Print per-dataset contamination statistics from score files."""
    paths = get_paths(period)
    LAB_scores_dir = paths["LAB_scores_dir"]
    end_year = paths["end_year"]

    print(f"\nTemporal Contamination Report (post-{end_year} knowledge)")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'Total':>8} {'Keep':>8} {'Remove':>8} {'% Remove':>10}")
    print(f"{'-'*70}")

    grand_total = 0
    grand_keep = 0
    grand_remove = 0

    for dataset_name in ALL_DATASETS:
        score_path = LAB_scores_dir / f"{dataset_name}_scores.jsonl"
        input_path = INSTRUCT_DIR / f"{dataset_name}.jsonl"

        if not score_path.exists():
            if input_path.exists():
                total = sum(1 for l in open(input_path, "r", encoding="utf-8")
                            if l.strip())
                print(f"{dataset_name:<20} {total:>8,} {'--':>8} {'--':>8} "
                      f"{'NOT SCORED':>10}")
            else:
                print(f"{dataset_name:<20} {'--':>8} {'--':>8} {'--':>8} "
                      f"{'NOT FOUND':>10}")
            continue

        # Read scores
        total = 0
        keep_count = 0
        remove_count = 0
        with open(score_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    score = json.loads(line)
                    total += 1
                    if score["keep"]:
                        keep_count += 1
                    else:
                        remove_count += 1

        pct = (remove_count / total * 100) if total > 0 else 0
        print(f"{dataset_name:<20} {total:>8,} {keep_count:>8,} "
              f"{remove_count:>8,} {pct:>9.1f}%")

        grand_total += total
        grand_keep += keep_count
        grand_remove += remove_count

    print(f"{'-'*70}")
    grand_pct = (grand_remove / grand_total * 100) if grand_total > 0 else 0
    print(f"{'TOTAL':<20} {grand_total:>8,} {grand_keep:>8,} "
          f"{grand_remove:>8,} {grand_pct:>9.1f}%")


# ---------------------------------------------------------------------------
# Filter datasets based on scores
# ---------------------------------------------------------------------------

def filter_datasets(period):
    """Create filtered JSONL files based on scores."""
    paths = get_paths(period)
    LAB_scores_dir = paths["LAB_scores_dir"]
    output_dir = paths["posttraining_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in ALL_DATASETS:
        score_path = LAB_scores_dir / f"{dataset_name}_scores.jsonl"
        input_path = INSTRUCT_DIR / f"{dataset_name}.jsonl"

        if not score_path.exists():
            print(f"  {dataset_name}: no scores found, skipping")
            continue
        if not input_path.exists():
            print(f"  {dataset_name}: no input data found, skipping")
            continue

        # Read scores
        scores = []
        with open(score_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    scores.append(json.loads(line))

        # Read conversations
        conversations = read_jsonl(str(input_path))

        if len(scores) != len(conversations):
            print(f"  {dataset_name}: WARNING score count ({len(scores)}) != "
                  f"data count ({len(conversations)}), skipping")
            continue

        # Filter
        filtered = []
        removed_items = []
        for score, conv in zip(scores, conversations):
            if score["keep"]:
                filtered.append(conv)
            else:
                removed_items.append(conv)

        # Write filtered dataset
        output_path = output_dir / f"{dataset_name}_filtered.jsonl"
        write_jsonl(filtered, str(output_path), validate=False)

        # Write removed items for inspection
        removed_path = output_dir / f"{dataset_name}_removed.jsonl"
        if removed_items:
            with open(removed_path, "w", encoding="utf-8") as f:
                for conv in removed_items:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        pct = len(removed_items) / len(conversations) * 100 if conversations else 0
        print(f"  {dataset_name}: kept {len(filtered):,}/{len(conversations):,} "
              f"(removed {len(removed_items):,}, {pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze temporal contamination scores and filter datasets"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--stats", action="store_true",
                        help="Print per-dataset contamination statistics")
    parser.add_argument("--filter", action="store_true",
                        help="Create filtered JSONL files in posttraining_data/")
    parser.add_argument("--import-batch", action="store_true",
                        help="Import scores from existing Batch API results")
    args = parser.parse_args()

    paths = get_paths(args.period)
    print(f"Period: {args.period} ({paths['start_year']}-{paths['end_year']})")

    if args.import_batch:
        print(f"\nImporting batch results from: {paths['posttraining_dir'] / 'batch_temp'}")
        import_batch_scores(args.period)

    if args.stats:
        print_stats(args.period)

    if args.filter:
        print(f"\nFiltering datasets to: {paths['posttraining_dir']}")
        filter_datasets(args.period)

    if not (args.stats or args.filter or args.import_batch):
        print("\nChoose an action:")
        print("  --import-batch : Import scores from existing Batch API results")
        print("  --stats        : Print per-dataset contamination statistics")
        print("  --filter       : Create filtered JSONL files")
