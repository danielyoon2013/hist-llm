"""
Direct QA/CoT generation without synthetic-data-kit dependency.

Replaces run.py's subprocess calls to synthetic-data-kit CLI with direct
OpenAI API calls using ThreadPoolExecutor for high concurrency.
~5-10x faster than the subprocess-based approach.

Output format is identical to synthetic-data-kit, so convert.py works unchanged.

Usage:
    # Generate QA + CoT for one collection
    python -m src.post_training.corpus.run_direct --period 1950_1999 --collection nyt_filtered

    # QA only, custom concurrency
    python -m src.post_training.corpus.run_direct --period 1950_1999 --collection ft --skip-cot --max-workers 80

    # Multiple collections at once
    python -m src.post_training.corpus.run_direct --period 1950_1999 --collection nyt_filtered economist ft newswire

    # Then convert (unchanged):
    python -m src.post_training.corpus.convert --period 1950_1999
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.post_training.config import PERIODS, get_paths, load_api_key, MODEL, MAX_RETRIES, RETRY_BASE_DELAY


# ---------------------------------------------------------------------------
# Prompts (from synth_config.yaml, adapted for JSON object response format)
# ---------------------------------------------------------------------------

QA_PROMPT = """Create {num_pairs} question-answer pairs from this text for LLM training.

Rules:
1. Questions must require analytical thinking, not just fact lookup
2. Answers must be directly supported by the text
3. Vary question types: cause-effect, comparison, analysis, inference, summary
4. Return a JSON object with key "qa_pairs" containing an array:

{{"qa_pairs": [{{"question": "Question 1?", "answer": "Answer 1."}}, {{"question": "Question 2?", "answer": "Answer 2."}}]}}

Text:
{text}"""

COT_PROMPT = """Create {num_cot} complex reasoning examples from this text that demonstrate chain-of-thought thinking.

Each example should have:
1. A challenging question that requires step-by-step reasoning
2. Detailed reasoning steps that break down the problem
3. A concise final answer

Return a JSON object with key "cot_examples" containing an array:

{{"cot_examples": [{{"question": "Complex question?", "reasoning": "Step 1: First, I need to consider...\\nStep 2: Then, I analyze...\\nStep 3: Finally, I can conclude...", "answer": "Final answer based on the reasoning."}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Text chunking (replicates synthetic-data-kit's chunking logic)
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=6000, overlap=300):
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# API call with retry (thread-safe, uses shared client)
# ---------------------------------------------------------------------------

def _call_api(client, prompt, model=MODEL, max_tokens=4096, temperature=0.7):
    """Call OpenAI API with retry logic. Returns parsed JSON dict or None."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_DELAY * (attempt + 1)  # linear backoff: 2, 4, 6, 8, 10s
                print(f"  [retry] {type(e).__name__}: {e} (attempt {attempt+1}/{MAX_RETRIES}, wait {wait}s)",
                      flush=True)
                time.sleep(wait)
            else:
                print(f"  [FAILED] {type(e).__name__}: {e} (exhausted {MAX_RETRIES} retries)",
                      flush=True)
                return None


# ---------------------------------------------------------------------------
# Task processor (runs in thread pool)
# ---------------------------------------------------------------------------

def process_task(task):
    """Process a single QA or CoT generation task. Thread worker function."""
    client, task_type, doc_name, chunk_idx, chunk, num_items, model = task

    if task_type == "qa":
        prompt = QA_PROMPT.format(num_pairs=num_items, text=chunk)
        result = _call_api(client, prompt, model=model)
        items = result.get("qa_pairs", []) if result else []
    else:
        prompt = COT_PROMPT.format(num_cot=num_items, text=chunk)
        result = _call_api(client, prompt, model=model)
        items = result.get("cot_examples", []) if result else []

    return (doc_name, task_type, chunk_idx, items)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_direct(period, collections, max_workers=50, num_qa=3, num_cot=2,
               skip_cot=False, chunk_size=6000, overlap=300):
    """Run QA/CoT generation directly with concurrent API calls."""
    from openai import OpenAI

    paths = get_paths(period)
    synthetic_dir = paths["synthetic_dir"]
    input_base = synthetic_dir / "input"
    generated_base = synthetic_dir / "generated"

    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    mode = "QA + CoT" if not skip_cot else "QA only"
    print(f"Period: {period} ({paths['start_year']}-{paths['end_year']})")
    print(f"Mode: {mode}")
    print(f"Max workers: {max_workers}")
    print(f"Model: {MODEL}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")

    for collection in collections:
        # Try parquet first (fast), fall back to .txt directory
        parquet_path = input_base / f"{collection}.parquet"
        collection_dir = input_base / collection

        if parquet_path.exists():
            source = "parquet"
        elif collection_dir.exists():
            source = "txt"
        else:
            print(f"\nError: Neither {parquet_path} nor {collection_dir}/ found, skipping")
            continue

        generated_dir = generated_base / collection
        os.makedirs(generated_dir, exist_ok=True)

        print(f"\n{'=' * 70}")

        # Read documents
        if source == "parquet":
            print(f"Reading {parquet_path.name}...", flush=True)
            df = pd.read_parquet(parquet_path)
            docs = [(row["doc_name"], row["text"]) for _, row in df.iterrows()
                    if row["text"] and str(row["text"]).strip()]
            num_docs = len(docs)
            print(f"Collection: {collection} ({num_docs} documents from parquet)",
                  flush=True)
        else:
            print(f"Reading .txt files from {collection_dir}/...", flush=True)
            txt_files = sorted(collection_dir.glob("*.txt"))
            docs = []
            for i, txt_file in enumerate(txt_files):
                text = txt_file.read_text(encoding="utf-8")
                if text.strip():
                    docs.append((txt_file.stem, text))
                if (i + 1) % 1000 == 0:
                    print(f"  Read {i + 1}/{len(txt_files)} files...", flush=True)
            num_docs = len(docs)
            print(f"Collection: {collection} ({num_docs} documents from .txt)",
                  flush=True)

        print(f"Output: {generated_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Build task list
        tasks = []
        doc_chunk_counts = {}

        for doc_name, text in docs:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            doc_chunk_counts[doc_name] = len(chunks)

            for i, chunk in enumerate(chunks):
                tasks.append((client, "qa", doc_name, i, chunk, num_qa, MODEL))
                if not skip_cot:
                    tasks.append((client, "cot", doc_name, i, chunk, num_cot, MODEL))

        total_tasks = len(tasks)
        avg_chunks = (sum(doc_chunk_counts.values()) / len(doc_chunk_counts)
                      if doc_chunk_counts else 0)
        print(f"Tasks: {total_tasks:,} "
              f"({len(doc_chunk_counts)} docs, avg {avg_chunks:.1f} chunks/doc)",
              flush=True)

        # Process all tasks concurrently
        results = {}  # (doc_name, type) -> list of items
        completed = 0
        failed = 0
        start_time = time.time()
        last_print_time = start_time

        print(f"Submitting to thread pool ({max_workers} workers)...", flush=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            print(f"All {total_tasks:,} tasks submitted. Waiting for completions...",
                  flush=True)

            for future in as_completed(futures):
                try:
                    doc_name, task_type, chunk_idx, items = future.result()
                    key = (doc_name, task_type)
                    if key not in results:
                        results[key] = []
                    results[key].extend(items)
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"  [ERROR] Task exception: {type(e).__name__}: {e}",
                          flush=True)

                # Progress every 25 tasks or every 10 seconds
                done = completed + failed
                now = time.time()
                if done % 25 == 0 or done == total_tasks or (now - last_print_time) >= 10:
                    last_print_time = now
                    elapsed = now - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (total_tasks - done) / rate if rate > 0 else 0
                    eta_time = datetime.now() + timedelta(seconds=remaining)
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"{done:,}/{total_tasks:,} tasks "
                          f"({100 * done / total_tasks:.1f}%) | "
                          f"Rate: {rate:.1f} tasks/s | "
                          f"ETA: {remaining / 60:.1f}m ({eta_time.strftime('%H:%M')}) | "
                          f"OK: {completed} Failed: {failed}",
                          flush=True)

        elapsed = time.time() - start_time

        # Convert results directly to nanochat JSONL (single file, fast I/O)
        total_qa_pairs = 0
        total_cot_examples = 0
        conversations = []

        for (doc_name, task_type), items in results.items():
            if not items:
                continue

            for item in items:
                if not item.get("question") or not item.get("answer"):
                    continue

                if task_type == "cot":
                    # Wrap reasoning in <think> tags
                    answer = item.get("answer", "")
                    reasoning = item.get("reasoning", "")
                    if reasoning:
                        answer = f"<think>\n{reasoning}\n</think>\n{answer}"
                    conv = [
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": answer},
                    ]
                    total_cot_examples += 1
                else:
                    conv = [
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]},
                    ]
                    total_qa_pairs += 1

                conversations.append(conv)

        # Write single JSONL per collection
        jsonl_path = generated_dir / f"{collection}_qa_cot.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        print(f"\n  Complete in {elapsed:.1f}s ({elapsed / 60:.1f}m)", flush=True)
        print(f"  QA pairs: {total_qa_pairs:,}", flush=True)
        if not skip_cot:
            print(f"  CoT examples: {total_cot_examples:,}", flush=True)
        print(f"  Total conversations: {len(conversations):,}", flush=True)
        print(f"  Failed tasks: {failed}", flush=True)
        print(f"  Output: {jsonl_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct QA/CoT generation (no synthetic-data-kit dependency)"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--collection", type=str, nargs="+", required=True,
                        help="Collection(s) to process")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Concurrent API calls (default: 50)")
    parser.add_argument("--num-qa", type=int, default=3,
                        help="QA pairs per chunk (default: 3)")
    parser.add_argument("--num-cot", type=int, default=2,
                        help="CoT examples per chunk (default: 2)")
    parser.add_argument("--skip-cot", action="store_true",
                        help="Skip CoT generation (QA only)")
    parser.add_argument("--chunk-size", type=int, default=6000,
                        help="Characters per chunk (default: 6000)")
    parser.add_argument("--overlap", type=int, default=300,
                        help="Overlap between chunks (default: 300)")
    args = parser.parse_args()

    run_direct(
        args.period,
        args.collection,
        max_workers=args.max_workers,
        num_qa=args.num_qa,
        num_cot=args.num_cot,
        skip_cot=args.skip_cot,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
