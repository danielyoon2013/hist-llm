"""
Filter datasets for temporal contamination using OpenAI Batch API.

For each conversation, asks GPT-4o-mini whether it requires knowledge that
didn't exist before the period's end year. Keeps only clean examples.

Supports both external instruct datasets (SmolTalk, MMLU, ARC) and
corpus Q&A generated from historical documents.

Usage:
    # Filter external instruct datasets
    python -m src.post_training.instruct.filter --period 1950_1999 --submit
    python -m src.post_training.instruct.filter --period 1950_1999 --check
    python -m src.post_training.instruct.filter --period 1950_1999 --process

    # Filter corpus Q&A (add --corpus flag)
    python -m src.post_training.instruct.filter --period 1950_1999 --submit --corpus
    python -m src.post_training.instruct.filter --period 1950_1999 --check --corpus
    python -m src.post_training.instruct.filter --period 1950_1999 --process --corpus

    # Dry run (test 5 examples)
    python -m src.post_training.instruct.filter --period 1950_1999 --dry-run
    python -m src.post_training.instruct.filter --period 1950_1999 --dry-run --corpus
"""

import os
import json
import argparse
from pathlib import Path

from src.post_training.config import PERIODS, get_paths, PROJECT_ROOT
from src.post_training.utils import (
    call_openai_json, read_jsonl, write_jsonl,
    create_batch_request_file, submit_batch,
    check_batch_status, download_batch_results,
)


INSTRUCT_DIR = PROJECT_ROOT / "data" / "instruct_data"

# External instruct datasets (GSM8K excluded — pure math, no temporal content)
INSTRUCT_DATASETS = ["smoltalk", "mmlu", "arc_easy", "arc_challenge"]

# OpenAI Batch API limit: 50,000 requests per batch
BATCH_CHUNK_SIZE = 50000


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------

def get_datasets(paths, corpus_only=False):
    """
    Return list of (dataset_name, input_path) tuples to process.

    Args:
        paths: Period paths from get_paths()
        corpus_only: If True, return only corpus Q&A. If False, return instruct datasets.
    """
    if corpus_only:
        return [("corpus", paths["hist_corpus_qa_output"])]
    else:
        return [(name, INSTRUCT_DIR / f"{name}.jsonl") for name in INSTRUCT_DATASETS]


# ---------------------------------------------------------------------------
# Text extraction and prompt building
# ---------------------------------------------------------------------------

def extract_text_for_classification(messages):
    """
    Extract the key text from a conversation for temporal classification.
    Uses the first user message + first assistant message to keep costs low.
    """
    parts = []
    for msg in messages[:4]:  # first 2 turns max
        parts.append(f"[{msg['role']}]: {msg['content'][:500]}")
    return "\n".join(parts)


def build_classification_prompt(text, end_year):
    return f"""Does the following conversation require knowledge that did NOT exist before the year {end_year}?

Consider: historical events, technology, people who became famous, scientific discoveries, cultural works, organizations, and terminology that emerged after {end_year}.

Conversation:
---
{text}
---

Respond with JSON: {{"keep": true}} if the conversation only uses knowledge available before {end_year}, or {{"keep": false}} if it requires post-{end_year} knowledge."""


# ---------------------------------------------------------------------------
# Dry run: classify a few examples in real-time
# ---------------------------------------------------------------------------

def dry_run(period, corpus_only=False):
    paths = get_paths(period)
    end_year = paths["end_year"]
    datasets = get_datasets(paths, corpus_only)

    for dataset_name, input_path in datasets:
        if not input_path.exists():
            print(f"  {dataset_name}: not found at {input_path}, skipping")
            continue

        conversations = read_jsonl(str(input_path))
        sample = conversations[:5]
        print(f"\n--- {dataset_name} ({len(conversations):,} total) ---")

        for i, msgs in enumerate(sample):
            text = extract_text_for_classification(msgs)
            prompt = build_classification_prompt(text, end_year)
            result = call_openai_json([{"role": "user", "content": prompt}], max_tokens=64)
            keep = result.get("keep", None)
            status = "KEEP" if keep else "REMOVE"
            preview = msgs[0]["content"][:100] if msgs else ""
            print(f"  [{status}] {preview}...")


# ---------------------------------------------------------------------------
# Step 1: Create and submit batch requests
# ---------------------------------------------------------------------------

def submit_filter_batch(period, corpus_only=False):
    paths = get_paths(period)
    end_year = paths["end_year"]
    batch_dir = paths["posttraining_dir"] / "batch_temp"
    os.makedirs(batch_dir, exist_ok=True)

    datasets = get_datasets(paths, corpus_only)

    for dataset_name, input_path in datasets:
        if not input_path.exists():
            print(f"  {dataset_name}: not found at {input_path}, skipping")
            continue

        conversations = read_jsonl(str(input_path))
        print(f"\n{dataset_name}: {len(conversations):,} rows")

        # Build batch requests
        requests = []
        for i, msgs in enumerate(conversations):
            text = extract_text_for_classification(msgs)
            prompt = build_classification_prompt(text, end_year)
            requests.append({
                "custom_id": f"{dataset_name}_{i}",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
            })

        # Split into chunks of BATCH_CHUNK_SIZE (OpenAI limit: 50K per batch)
        num_chunks = (len(requests) + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE
        batch_ids = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * BATCH_CHUNK_SIZE
            end = min(start + BATCH_CHUNK_SIZE, len(requests))
            chunk = requests[start:end]

            suffix = f"_chunk{chunk_idx}" if num_chunks > 1 else ""
            request_file = batch_dir / f"{dataset_name}{suffix}_requests.jsonl"
            create_batch_request_file(chunk, str(request_file))

            batch_id = submit_batch(
                str(request_file),
                description=f"filter_{dataset_name}{suffix}_{period}",
            )
            batch_ids.append(batch_id)
            print(f"  Chunk {chunk_idx+1}/{num_chunks}: {len(chunk):,} requests, Batch ID: {batch_id}")

        # Save all batch IDs (one per line) for later retrieval
        id_file = batch_dir / f"{dataset_name}_batch_ids.txt"
        with open(id_file, "w") as f:
            f.write("\n".join(batch_ids))
        print(f"  {num_chunks} batch(es) saved to {id_file}")


# ---------------------------------------------------------------------------
# Step 2: Check batch status
# ---------------------------------------------------------------------------

def check_batches(period, corpus_only=False):
    paths = get_paths(period)
    batch_dir = paths["posttraining_dir"] / "batch_temp"
    datasets = get_datasets(paths, corpus_only)

    for dataset_name, _ in datasets:
        id_file = batch_dir / f"{dataset_name}_batch_ids.txt"
        if not id_file.exists():
            print(f"\n{dataset_name}: no batch submitted yet")
            continue
        batch_ids = [line.strip() for line in open(id_file) if line.strip()]
        print(f"\n{dataset_name} ({len(batch_ids)} batch(es)):")
        for i, batch_id in enumerate(batch_ids):
            if len(batch_ids) > 1:
                print(f"  Chunk {i+1}/{len(batch_ids)}:")
            check_batch_status(batch_id)


# ---------------------------------------------------------------------------
# Step 3: Download results and filter
# ---------------------------------------------------------------------------

def process_batch_results(period, corpus_only=False):
    paths = get_paths(period)
    batch_dir = paths["posttraining_dir"] / "batch_temp"
    filtered_dir = paths["final_filtered_dir"]
    removed_dir = paths["final_removed_dir"]
    scores_dir = paths["LAB_scores_dir"]
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(removed_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)

    datasets = get_datasets(paths, corpus_only)

    for dataset_name, input_path in datasets:
        id_file = batch_dir / f"{dataset_name}_batch_ids.txt"
        if not id_file.exists():
            print(f"  {dataset_name}: no batch IDs found, skipping")
            continue

        batch_ids = [line.strip() for line in open(id_file) if line.strip()]

        # Download and merge results from all chunks
        all_results = []
        all_complete = True
        for i, batch_id in enumerate(batch_ids):
            suffix = f"_chunk{i}" if len(batch_ids) > 1 else ""
            results_file = batch_dir / f"{dataset_name}{suffix}_results.jsonl"
            results = download_batch_results(batch_id, str(results_file))
            if results is None:
                print(f"  {dataset_name}: chunk {i+1}/{len(batch_ids)} not complete yet")
                all_complete = False
                break
            all_results.extend(results)

        if not all_complete:
            continue
        results = all_results

        # Load original conversations
        conversations = read_jsonl(str(input_path))

        # Build a map from custom_id to keep/remove decision
        decisions = {}
        for custom_id, response_text in results:
            try:
                parsed = json.loads(response_text)
                decisions[custom_id] = parsed.get("keep", True)
            except json.JSONDecodeError:
                decisions[custom_id] = True  # keep on parse error (conservative)

        # Save scores to LAB_scores directory
        scores_path = scores_dir / f"{dataset_name}_scores.jsonl"
        with open(scores_path, "w", encoding="utf-8") as f:
            for custom_id, keep in decisions.items():
                f.write(json.dumps({"id": custom_id, "keep": keep}) + "\n")
        print(f"  Scores saved to: {scores_path}")

        # Filter conversations, logging removed items for sanity checking
        filtered = []
        removed_items = []
        for i, msgs in enumerate(conversations):
            key = f"{dataset_name}_{i}"
            if decisions.get(key, True):
                filtered.append(msgs)
            else:
                removed_items.append({"index": i, "messages": msgs})

        # Save filtered dataset
        output_path = filtered_dir / f"{dataset_name}_filtered.jsonl"
        write_jsonl(filtered, str(output_path))

        # Save removed items for inspection
        removed_path = removed_dir / f"{dataset_name}_removed.jsonl"
        with open(removed_path, "w", encoding="utf-8") as f:
            for item in removed_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        removed = len(removed_items)
        print(f"  {dataset_name}: kept {len(filtered):,}/{len(conversations):,} "
              f"(removed {removed:,}, {removed/len(conversations)*100:.1f}%)")
        print(f"    Output: {output_path}")
        print(f"    Removed: {removed_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter datasets for temporal contamination (LAB filtering)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter external instruct datasets (SmolTalk, MMLU, ARC)
  python -m src.post_training.instruct.filter --period 1950_1999 --submit
  python -m src.post_training.instruct.filter --period 1950_1999 --process

  # Filter corpus Q&A generated from historical documents
  python -m src.post_training.instruct.filter --period 1950_1999 --submit --corpus
  python -m src.post_training.instruct.filter --period 1950_1999 --process --corpus
"""
    )
    parser.add_argument("--period", type=str, required=True, choices=list(PERIODS.keys()))
    parser.add_argument("--corpus", action="store_true",
                        help="Filter corpus Q&A instead of external instruct datasets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test classify 5 examples in real-time")
    parser.add_argument("--submit", action="store_true",
                        help="Step 1: Submit batch jobs to OpenAI (~24h)")
    parser.add_argument("--check", action="store_true",
                        help="Step 2: Check batch job status")
    parser.add_argument("--process", action="store_true",
                        help="Step 3: Download results and save filtered datasets")
    args = parser.parse_args()

    paths = get_paths(args.period)
    dataset_type = "corpus Q&A" if args.corpus else "instruct datasets"

    print(f"Period: {args.period} ({paths['start_year']}-{paths['end_year']})")
    print(f"Dataset type: {dataset_type}")
    if args.corpus:
        print(f"Input: {paths['hist_corpus_qa_output']}")
    else:
        print(f"Input dir: {INSTRUCT_DIR}")
    print(f"Filtered output dir: {paths['final_filtered_dir']}")
    print(f"Removed output dir: {paths['final_removed_dir']}")

    if args.dry_run:
        dry_run(args.period, corpus_only=args.corpus)
    elif args.submit:
        submit_filter_batch(args.period, corpus_only=args.corpus)
    elif args.check:
        check_batches(args.period, corpus_only=args.corpus)
    elif args.process:
        process_batch_results(args.period, corpus_only=args.corpus)
    else:
        print("\nChoose an action:")
        print("  --dry-run   : Test classify 5 examples in real-time")
        print("  --submit    : Submit batch filtering jobs to OpenAI (~24h)")
        print("  --check     : Check batch job status")
        print("  --process   : Download results and save filtered datasets")
        print("\nAdd --corpus to filter corpus Q&A instead of instruct datasets")
