"""
Convert synthetic-data-kit output to nanochat CustomJSON format.

Reads QA pairs and CoT examples from the toolkit's generated/curated output
and converts them to nanochat's conversation format:
  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

Writes per-collection files to final/filtered/:
  hist_{collection}.jsonl  (one file per collection)

Supports both flat structure and per-collection subdirectories:
  synthetic/curated/*.json           (flat)
  synthetic/curated/{collection}/*.json  (per-collection)

Usage:
    python -m src.post_training.corpus.convert --period 1950_1999
    python -m src.post_training.corpus.convert --period 1950_1999 --no-cot
"""

import os
import json
import argparse
from pathlib import Path

from src.post_training.config import PERIODS, get_paths, PROJECT_ROOT
from src.post_training.utils import validate_conversation, write_jsonl


def load_qa_pairs(json_path):
    """Load QA pairs from a toolkit output JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    if isinstance(data, dict):
        if "qa_pairs" in data:
            pairs.extend(data["qa_pairs"])
        if "filtered_pairs" in data:
            pairs.extend(data["filtered_pairs"])
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                pairs.append(item)

    return pairs


def load_cot_examples(json_path):
    """Load CoT examples from a toolkit output JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    if isinstance(data, dict) and "cot_examples" in data:
        for ex in data["cot_examples"]:
            answer = ex.get("answer", "")
            reasoning = ex.get("reasoning", "")
            if reasoning:
                answer = f"<think>\n{reasoning}\n</think>\n{answer}"
            pairs.append({
                "question": ex.get("question", ""),
                "answer": answer,
            })
    return pairs


def convert_pair_to_conversation(pair):
    """Convert a single QA pair dict to nanochat conversation format."""
    return [
        {"role": "user", "content": pair["question"]},
        {"role": "assistant", "content": pair["answer"]},
    ]


def find_json_files(base_dir, pattern):
    """Find JSON files matching pattern, searching subdirectories too."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    # First try flat structure
    flat_files = list(base_dir.glob(pattern))

    # Then search subdirectories (per-collection structure)
    subdir_files = list(base_dir.glob(f"*/{pattern}"))

    return sorted(set(flat_files + subdir_files))


def convert_pipeline_output(period, source_dir=None, cot_dir=None, include_cot=True):
    """
    Convert toolkit output files to per-collection nanochat JSONL files.

    Writes one file per collection to final/filtered/:
      hist_{collection}.jsonl

    Fast path: checks for pre-built JSONL files from run_direct.py first.
    Slow path: falls back to reading individual JSON files from synthetic-data-kit.

    Args:
        period: Period key (e.g., "1950_1999")
        source_dir: Directory with curated QA JSON files (auto-detected if None)
        cot_dir: Directory with CoT JSON files (auto-detected if None)
        include_cot: Whether to include CoT examples from generated/
    """
    paths = get_paths(period)
    synthetic_dir = paths["synthetic_dir"]
    generated_dir = synthetic_dir / "generated"
    output_dir = paths["final_filtered_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Per-collection accumulator: {collection_name: [conversations]}
    collection_conversations = {}

    def add_conversation(collection, conv):
        if collection not in collection_conversations:
            collection_conversations[collection] = []
        collection_conversations[collection].append(conv)

    # --- Fast path: check for JSONL files from run_direct.py ---
    jsonl_files = []
    if generated_dir.exists():
        # Check for collection-level JSONL files (from run_direct.py)
        for subdir in sorted(generated_dir.iterdir()):
            if subdir.is_dir():
                for jf in subdir.glob("*_qa_cot.jsonl"):
                    jsonl_files.append(jf)
        # Also check flat JSONL files
        for jf in sorted(generated_dir.glob("*_qa_cot.jsonl")):
            jsonl_files.append(jf)

    if jsonl_files:
        print(f"Found {len(jsonl_files)} pre-built JSONL file(s) (fast path):")
        for jf in jsonl_files:
            count = 0
            collection_name = jf.stem.replace("_qa_cot", "")
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        conv = json.loads(line)
                        valid, msg = validate_conversation(conv)
                        if valid:
                            add_conversation(collection_name, conv)
                            count += 1
            print(f"  {jf.name}: {count:,} conversations")

    # --- Slow path: individual JSON files (existing collections from run.py) ---
    # Auto-detect source directories
    if source_dir is None:
        candidates = [
            generated_dir,
            synthetic_dir / "curated",
        ]
        for c in candidates:
            if c.exists():
                files = find_json_files(c, "*_qa_*.json")
                if files:
                    source_dir = c
                    break

    if source_dir is not None:
        source_dir = Path(source_dir)
        qa_files = find_json_files(source_dir, "*_qa_*.json")

        if qa_files:
            print(f"\nReading {len(qa_files):,} individual JSON files (slow path)...")
            skipped = 0
            for i, json_file in enumerate(qa_files):
                rel_path = json_file.relative_to(source_dir)
                collection = rel_path.parts[0] if len(rel_path.parts) > 1 else "_flat"

                pairs = load_qa_pairs(json_file)
                for pair in pairs:
                    if not pair.get("question") or not pair.get("answer"):
                        skipped += 1
                        continue
                    conv = convert_pair_to_conversation(pair)
                    valid, msg = validate_conversation(conv)
                    if valid:
                        add_conversation(collection, conv)
                    else:
                        skipped += 1

                if (i + 1) % 5000 == 0:
                    print(f"  Read {i + 1:,}/{len(qa_files):,} QA files...", flush=True)

            # CoT from individual JSON files
            if include_cot:
                cot_source = cot_dir or source_dir
                if cot_source:
                    cot_source = Path(cot_source)
                    cot_files = find_json_files(cot_source, "*_cot_*.json")
                    if cot_files:
                        print(f"  Reading {len(cot_files):,} CoT files...")
                        for i, json_file in enumerate(cot_files):
                            rel_path = json_file.relative_to(cot_source)
                            collection = rel_path.parts[0] if len(rel_path.parts) > 1 else "_flat"

                            pairs = load_cot_examples(json_file)
                            for pair in pairs:
                                if not pair.get("question") or not pair.get("answer"):
                                    skipped += 1
                                    continue
                                conv = convert_pair_to_conversation(pair)
                                valid, msg = validate_conversation(conv)
                                if valid:
                                    add_conversation(collection, conv)
                                else:
                                    skipped += 1

                            if (i + 1) % 5000 == 0:
                                print(f"  Read {i + 1:,}/{len(cot_files):,} CoT files...",
                                      flush=True)

            print(f"  Skipped: {skipped}")

    if not collection_conversations:
        print("No conversations found. Run the pipeline first.")
        return

    # Write per-collection output files
    total_all = 0
    print(f"\nWriting per-collection files to {output_dir}:")
    for coll in sorted(collection_conversations.keys()):
        convs = collection_conversations[coll]
        output_path = output_dir / f"hist_{coll}.jsonl"
        write_jsonl(convs, str(output_path), validate=False)
        total_all += len(convs)
        print(f"  hist_{coll}.jsonl: {len(convs):,} conversations", flush=True)

    # Summary
    print(f"\nConversion complete:")
    print(f"  Collections: {len(collection_conversations)}")
    print(f"  Total conversations: {total_all:,}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert synthetic-data-kit output to nanochat format"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--source-dir", type=str, default=None,
                        help="Directory with curated QA files (auto-detected)")
    parser.add_argument("--cot-dir", type=str, default=None,
                        help="Directory with CoT files (auto-detected)")
    parser.add_argument("--no-cot", action="store_true",
                        help="Exclude CoT examples")
    args = parser.parse_args()

    paths = get_paths(args.period)
    print(f"Period: {args.period} ({paths['start_year']}-{paths['end_year']})")

    convert_pipeline_output(
        args.period,
        source_dir=args.source_dir,
        cot_dir=args.cot_dir,
        include_cot=not args.no_cot,
    )
