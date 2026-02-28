"""
Base class for all synthetic data generators.

Supports two execution modes:
- Batch API (submit/check/process): 50% cost savings, ~24h turnaround
- Sync API (--sync flag): ThreadPoolExecutor, instant results for testing

Common patterns:
- Text chunking (6000 chars, 300 overlap)
- JSONL output in nanochat CustomJSON format
- Multi-format rendering (MC-4, MC-2, Open, CoT, Passage variants)
"""

import os
import json
import time
import random as _random
import itertools
from collections import defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.post_training.config import PERIODS, get_paths, load_api_key, MODEL, MAX_RETRIES, RETRY_BASE_DELAY
from src.post_training.utils import validate_conversation, write_jsonl


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

# All supported format keys
FORMAT_MC4 = "mc4"                # 4-choice MC (MMLU, ARC, HellaSwag, LAB Eval)
FORMAT_MC2 = "mc2"                # 2-choice MC (PIQA, WinoGrande)
FORMAT_MC4_PASSAGE = "mc4_passage"  # 4-choice MC with passage (RACE)
FORMAT_MC2_PASSAGE = "mc2_passage"  # 2-choice MC with passage (BoolQ)
FORMAT_OPEN = "open"              # Open-ended Q&A (GSM8K)
FORMAT_COT = "cot"                # Chain-of-thought with <think> tags

MAX_PASSAGE_LENGTH = 2000  # Truncate passages for passage-based formats


# ---------------------------------------------------------------------------
# Format rendering utilities (matching nanochat/tasks/common.py exactly)
# ---------------------------------------------------------------------------

def render_mc(question, letters, choices):
    """Render MC question in nanochat format.

    Matches nanochat/tasks/common.py:render_mc() exactly:
    - choice text BEFORE letter (better token binding for small models)
    - No whitespace before letter (token ID consistency)
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


def make_mc_choices(correct, distractors, num_choices=4, position_idx=None):
    """Place correct answer among distractors with balanced positioning.

    Args:
        correct: The correct answer text.
        distractors: List of incorrect answer texts.
        num_choices: Total number of choices (4 for MC-4, 2 for MC-2).
        position_idx: Cyclic index for balanced placement. When provided,
            correct answer goes to position (position_idx % num_choices).
            This guarantees uniform distribution across A/B/C/D over a dataset.

    Returns (letters, ordered_choices, correct_letter).
    """
    wrong = list(distractors[:num_choices - 1])
    target_pos = position_idx % num_choices if position_idx is not None else 0
    pool = wrong[:target_pos] + [correct] + wrong[target_pos:]
    letters = tuple("ABCD"[:len(pool)])
    correct_letter = letters[target_pos]
    return letters, pool, correct_letter


def truncate_passage(text, max_len=MAX_PASSAGE_LENGTH):
    """Truncate passage for passage-based formats, breaking at sentence boundary."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_period = truncated.rfind('. ')
    if last_period > max_len // 2:
        return truncated[:last_period + 1]
    return truncated + "..."


# ---------------------------------------------------------------------------
# Shared utilities
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


def call_api(client, prompt, model=MODEL, max_tokens=4096, temperature=0.7):
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
                wait = RETRY_BASE_DELAY * (attempt + 1)
                print(f"  [retry] {type(e).__name__}: {e} "
                      f"(attempt {attempt+1}/{MAX_RETRIES}, wait {wait}s)", flush=True)
                time.sleep(wait)
            else:
                print(f"  [FAILED] {type(e).__name__}: {e} "
                      f"(exhausted {MAX_RETRIES} retries)", flush=True)
                return None


# ---------------------------------------------------------------------------
# Base generator
# ---------------------------------------------------------------------------

class BaseGenerator(ABC):
    """Abstract base class for synthetic data generators."""

    name: str = ""
    items_per_chunk: int = 3
    needs_corpus: bool = True   # False for D (temporal) and H (hist facts)
    num_batches: int = 10       # for metadata-based generators (D)
    train_only: bool = False    # True for H (factual recall — eval via external benchmarks)
    SUPPORTED_FORMATS: tuple = (FORMAT_OPEN,)  # override in subclasses

    @abstractmethod
    def build_prompt(self, chunk, period, start_year, end_year):
        """Build the generation prompt for a text chunk (or batch_num for metadata-based)."""
        ...

    @abstractmethod
    def parse_response(self, response):
        """Parse API response dict into list of raw items."""
        ...

    @abstractmethod
    def format_conversation(self, item, fmt, source_chunk=None):
        """Convert a raw item into nanochat conversation format for a given format.

        Args:
            item: Raw item dict from parse_response()
            fmt: Format string (e.g. FORMAT_MC4, FORMAT_OPEN, FORMAT_COT, etc.)
            source_chunk: Original text chunk (for passage-based formats)

        Returns:
            List of {role, content} dicts, or None if format not applicable.
        """
        ...

    def run(self, period, collections=None, max_workers=50, max_docs=None,
            chunk_size=6000, overlap=300, target_examples=None, action="run"):
        """Run this generator for a period, producing per-format output files.

        Args:
            target_examples: Target number of output examples for this generator.
                For metadata-based generators (D, H), this controls the number
                of API calls. For corpus-based generators, use max_docs instead.
            action: Execution mode:
                "submit"  — build prompts and submit OpenAI batch (no API key needed for check)
                "check"   — handled at generate.py level, returns None
                "process" — download batch results and write output files
                "run"     — legacy sync mode (ThreadPoolExecutor)

        Returns:
            For "submit": batch_id string or None.
            For "process"/"run": Dict of {format: Path} for output files, or None.
            For "check": None.
        """
        paths = get_paths(period)
        start_year, end_year = PERIODS[period]

        print(f"Generator: {self.name}")
        print(f"Period: {period} ({start_year}-{end_year})")
        print(f"Formats: {list(self.SUPPORTED_FORMATS)}")
        if target_examples:
            print(f"Target: {target_examples:,} examples")

        # --- Batch API dispatch ---
        if action == "submit":
            return self.submit_batch_requests(
                period, collections=collections, max_docs=max_docs,
                chunk_size=chunk_size, overlap=overlap,
                target_examples=target_examples,
            )
        elif action == "process":
            return self.process_batch_results(period)
        elif action == "check":
            return None  # handled at generate.py level

        # --- Sync mode (action="run") ---
        from openai import OpenAI

        generators_dir = paths["synthetic_dir"] / "by_generator"
        os.makedirs(generators_dir, exist_ok=True)

        api_key = load_api_key()
        client = OpenAI(api_key=api_key)

        print(f"Max workers: {max_workers}")
        print(f"Model: {MODEL}")

        # Build output paths per format
        output_paths = {
            fmt: generators_dir / f"{self.name}_{fmt}.jsonl"
            for fmt in self.SUPPORTED_FORMATS
        }

        # Per-format counters for balanced MC answer positioning.
        # Separate counter per format string prevents interleaving bias
        # when multiple MC formats share the same num_choices (e.g. mc4 + mc4_passage).
        # Thread-safe via GIL on next().
        self._mc_counters = defaultdict(itertools.count)

        if not self.needs_corpus:
            return self._run_metadata_based(
                client, period, start_year, end_year, max_workers, output_paths,
                target_examples=target_examples,
            )

        docs = self._load_documents(paths, collections, max_docs)
        if not docs:
            print("No documents found!")
            return None

        # Build task list
        tasks = []
        for doc_name, text in docs:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for i, chunk in enumerate(chunks):
                tasks.append((client, doc_name, i, chunk, period, start_year, end_year))

        total_tasks = len(tasks)
        print(f"Tasks: {total_tasks:,} ({len(docs)} docs)")
        for fmt, path in output_paths.items():
            print(f"  {fmt} -> {path.name}")

        # Process with ThreadPoolExecutor
        all_results = {fmt: [] for fmt in self.SUPPORTED_FORMATS}
        completed = 0
        failed = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_task, task): task
                for task in tasks
            }

            for future in as_completed(futures):
                try:
                    results = future.result()
                    for fmt, convs in results.items():
                        all_results[fmt].extend(convs)
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)

                done = completed + failed
                if done % 25 == 0 or done == total_tasks:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"{done:,}/{total_tasks:,} ({100*done/total_tasks:.1f}%) "
                          f"Rate: {rate:.1f}/s | OK: {completed} Failed: {failed}",
                          flush=True)

        # Write per-format output files
        elapsed = time.time() - start_time
        total_convs = 0
        for fmt in self.SUPPORTED_FORMATS:
            write_jsonl(all_results[fmt], str(output_paths[fmt]))
            count = len(all_results[fmt])
            total_convs += count
            print(f"  {fmt}: {count:,} conversations -> {output_paths[fmt].name}")

        print(f"Complete in {elapsed:.1f}s. "
              f"{total_convs:,} total conversations. Failed: {failed}")
        return output_paths

    def _process_task(self, task):
        """Process a single chunk. Returns {fmt: [conversations]}."""
        client, doc_name, chunk_idx, chunk, period, start_year, end_year = task

        prompt = self.build_prompt(chunk, period, start_year, end_year)
        response = call_api(client, prompt)

        if response is None:
            return {}

        items = self.parse_response(response)
        results = {}
        for item in items:
            for fmt in self.SUPPORTED_FORMATS:
                conv = self.format_conversation(item, fmt, source_chunk=chunk)
                if conv is None:
                    continue
                valid, err = validate_conversation(conv)
                if valid:
                    results.setdefault(fmt, []).append(conv)
        return results

    def _run_metadata_based(self, client, period, start_year, end_year,
                            max_workers, output_paths, target_examples=None):
        """Default implementation for metadata-based generators (D, H).

        Args:
            target_examples: If set, dynamically compute num_batches to hit
                this target. Otherwise uses self.num_batches (legacy default).
        """
        num_formats = len(self.SUPPORTED_FORMATS)
        if target_examples and num_formats > 0:
            # raw items needed = target / num_formats (each item → one conv per format)
            raw_needed = target_examples // num_formats
            self.num_batches = max(1, -(-raw_needed // self.items_per_chunk))  # ceil div

        print(f"Generating {self.num_batches} batches x {self.items_per_chunk} items")
        for fmt, path in output_paths.items():
            print(f"  {fmt} -> {path.name}")

        all_results = {fmt: [] for fmt in self.SUPPORTED_FORMATS}
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=min(max_workers, self.num_batches)) as executor:
            futures = {}
            for batch_num in range(1, self.num_batches + 1):
                prompt = self.build_prompt(batch_num, period, start_year, end_year)
                futures[executor.submit(call_api, client, prompt)] = batch_num

            for future in as_completed(futures):
                try:
                    response = future.result()
                    if response:
                        items = self.parse_response(response)
                        for item in items:
                            for fmt in self.SUPPORTED_FORMATS:
                                conv = self.format_conversation(item, fmt)
                                if conv is None:
                                    continue
                                valid, _ = validate_conversation(conv)
                                if valid:
                                    all_results[fmt].append(conv)
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)

        total_convs = 0
        for fmt in self.SUPPORTED_FORMATS:
            write_jsonl(all_results[fmt], str(output_paths[fmt]))
            count = len(all_results[fmt])
            total_convs += count
            print(f"  {fmt}: {count:,} conversations")

        print(f"Complete. {total_convs:,} total conversations. Failed: {failed}")
        return output_paths

    # ------------------------------------------------------------------
    # Batch API methods (submit / process)
    # ------------------------------------------------------------------

    def submit_batch_requests(self, period, collections=None, max_docs=None,
                              chunk_size=6000, overlap=300, target_examples=None):
        """Build prompts and submit as an OpenAI batch.

        Creates files in batch_temp/:
          {name}_requests.jsonl   — batch API request file
          {name}_manifest.jsonl   — custom_id -> chunk_text mapping
          {name}_batch_id.txt     — batch ID for retrieval

        Returns batch_id string, or None on failure.
        """
        from src.post_training.utils import create_batch_request_file, submit_batch

        paths = get_paths(period)
        start_year, end_year = PERIODS[period]
        batch_dir = paths["batch_temp_dir"]
        os.makedirs(batch_dir, exist_ok=True)

        # Build task list: [(custom_id, prompt_str, chunk_text_or_None), ...]
        if self.needs_corpus:
            docs = self._load_documents(paths, collections, max_docs)
            if not docs:
                print("No documents found!")
                return None
            tasks = []
            for doc_idx, (doc_name, text) in enumerate(docs):
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for chunk_idx, chk in enumerate(chunks):
                    cid = f"{self.name}_{doc_idx}_{chunk_idx}"
                    prompt = self.build_prompt(chk, period, start_year, end_year)
                    tasks.append((cid, prompt, chk))
        else:
            tasks = self._build_metadata_tasks(
                period, start_year, end_year, target_examples,
            )

        if not tasks:
            print("No tasks to submit!")
            return None

        total = len(tasks)
        print(f"Submitting {total:,} batch requests")

        # Write manifest (custom_id -> chunk_text for passage generators)
        manifest_path = batch_dir / f"{self.name}_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for cid, _, chunk_val in tasks:
                entry = {"custom_id": cid}
                if chunk_val is not None:
                    entry["chunk_text"] = chunk_val
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Write batch request file
        requests = []
        for cid, prompt, _ in tasks:
            requests.append({
                "custom_id": cid,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.7,
            })

        request_path = batch_dir / f"{self.name}_requests.jsonl"
        create_batch_request_file(requests, str(request_path))

        # Submit batch
        batch_id = submit_batch(
            str(request_path),
            description=f"{self.name}_{period}",
        )

        # Save batch ID
        id_path = batch_dir / f"{self.name}_batch_id.txt"
        with open(id_path, "w") as f:
            f.write(batch_id)
        print(f"  Batch ID saved: {id_path.name}")

        return batch_id

    def _build_metadata_tasks(self, period, start_year, end_year,
                              target_examples=None):
        """Build (custom_id, prompt, None) tuples for metadata generators.

        Default implementation for Gen D. Gen H overrides this.
        """
        num_formats = len(self.SUPPORTED_FORMATS)
        if target_examples and num_formats > 0:
            raw_needed = target_examples // num_formats
            num_batches = max(1, -(-raw_needed // self.items_per_chunk))
        else:
            num_batches = self.num_batches

        print(f"  {num_batches} metadata batches x {self.items_per_chunk} items/batch")

        tasks = []
        for batch_num in range(1, num_batches + 1):
            cid = f"{self.name}_batch_{batch_num}"
            prompt = self.build_prompt(batch_num, period, start_year, end_year)
            tasks.append((cid, prompt, None))
        return tasks

    def process_batch_results(self, period):
        """Download batch results, parse, format, validate, and write output.

        Reads batch_temp/{name}_batch_id.txt and {name}_manifest.jsonl.
        Writes by_generator/{name}_{fmt}.jsonl (identical output to sync mode).

        Returns dict of {format: Path} output files, or None if not ready.
        """
        from src.post_training.utils import download_batch_results

        paths = get_paths(period)
        batch_dir = paths["batch_temp_dir"]
        generators_dir = paths["synthetic_dir"] / "by_generator"
        os.makedirs(generators_dir, exist_ok=True)

        # Read batch ID
        id_path = batch_dir / f"{self.name}_batch_id.txt"
        if not id_path.exists():
            print(f"No batch ID found at {id_path}")
            return None
        batch_id = id_path.read_text().strip()

        # Download results
        results_path = batch_dir / f"{self.name}_results.jsonl"
        results = download_batch_results(batch_id, str(results_path))
        if results is None:
            print(f"Batch not complete yet")
            return None

        # Load manifest (custom_id -> chunk_text)
        manifest = {}
        manifest_path = batch_dir / f"{self.name}_manifest.jsonl"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    manifest[entry["custom_id"]] = entry.get("chunk_text")

        # Build output paths
        output_paths = {
            fmt: generators_dir / f"{self.name}_{fmt}.jsonl"
            for fmt in self.SUPPORTED_FORMATS
        }

        # Initialize MC counters
        self._mc_counters = defaultdict(itertools.count)

        # Process each result
        all_results = {fmt: [] for fmt in self.SUPPORTED_FORMATS}
        parsed_ok = 0
        parsed_fail = 0

        for custom_id, response_body in results:
            try:
                response = json.loads(response_body)
            except json.JSONDecodeError:
                parsed_fail += 1
                continue

            items = self.parse_response(response)
            chunk_val = manifest.get(custom_id)

            for item in items:
                for fmt in self.SUPPORTED_FORMATS:
                    conv = self.format_conversation(item, fmt, source_chunk=chunk_val)
                    if conv is None:
                        continue
                    valid, _ = validate_conversation(conv)
                    if valid:
                        all_results[fmt].append(conv)
            parsed_ok += 1

        # Write output files
        total_convs = 0
        for fmt in self.SUPPORTED_FORMATS:
            write_jsonl(all_results[fmt], str(output_paths[fmt]))
            count = len(all_results[fmt])
            total_convs += count
            print(f"  {fmt}: {count:,} conversations -> {output_paths[fmt].name}")

        print(f"Complete. {total_convs:,} total conversations. "
              f"Parsed: {parsed_ok:,} OK, {parsed_fail:,} failed")
        return output_paths

    def _load_documents(self, paths, collections, max_docs):
        """Load documents from synthetic/input/ directory."""
        import pandas as pd

        input_base = paths["synthetic_dir"] / "input"
        docs = []

        if collections is None:
            collections = []
            if input_base.exists():
                for f in sorted(input_base.iterdir()):
                    if f.suffix == ".parquet":
                        collections.append(f.stem)
                    elif f.is_dir() and not f.name.startswith("."):
                        collections.append(f.name)

        for collection in collections:
            parquet_path = input_base / f"{collection}.parquet"
            collection_dir = input_base / collection

            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                for _, row in df.iterrows():
                    if row.get("text") and str(row["text"]).strip():
                        docs.append((
                            f"{collection}/{row.get('doc_name', 'unknown')}",
                            row["text"],
                        ))
            elif collection_dir.exists():
                for txt_file in sorted(collection_dir.glob("*.txt")):
                    text = txt_file.read_text(encoding="utf-8")
                    if text.strip():
                        docs.append((f"{collection}/{txt_file.stem}", text))

        if max_docs and len(docs) > max_docs:
            import random
            rng = random.Random(42)
            docs = rng.sample(docs, min(max_docs, len(docs)))

        print(f"Loaded {len(docs)} documents from {len(collections)} collection(s)")
        return docs
