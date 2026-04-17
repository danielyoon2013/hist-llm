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
import re
import json
import time
import random as _random
import itertools
from collections import defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.post_training.config import (
    PERIODS, get_paths, load_api_key, MODEL, MAX_RETRIES, RETRY_BASE_DELAY,
    GENERATOR_SPEC, ITEMS_PER_CALL, GENERATOR_MODEL_OVERRIDES,
)
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
# OCR text cleaning
# ---------------------------------------------------------------------------

def clean_ocr_text(text):
    """Clean common OCR artifacts from historical document text.

    Fixes:
    - Hyphenated word breaks: 'respon- sible' → 'responsible'
    - Excessive internal whitespace
    """
    # Rejoin words broken by hyphen + whitespace + lowercase continuation.
    # Targets OCR line-break artifacts (e.g. "produc- tion", "Repub- lican").
    # Safe: real hyphens ("post-armistice") have no space after the hyphen.
    text = re.sub(r'(\w)- +([a-z])', r'\1\2', text)
    # Collapse runs of whitespace (spaces/tabs) to single space
    text = re.sub(r'[^\S\n]{2,}', ' ', text)
    return text


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

def chunk_text(text, chunk_size=6000, overlap=300, max_chunks_per_doc=3):
    """Split text into overlapping chunks.

    Args:
        max_chunks_per_doc: Cap on chunks per document. Prevents long docs
            (e.g. 250-chunk OpenAlex papers) from dominating the training set.
            Default 3 distributes generation across more unique source docs.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks_per_doc:
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def call_api(client, prompt, model=MODEL, max_tokens=4096, temperature=0.7, plaintext=False):
    """Call OpenAI API with retry logic.

    Returns parsed JSON dict by default, or a plain string when plaintext=True
    (used for Gen G rephrase prompts that produce raw text, not JSON).
    """
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if not plaintext:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return content if plaintext else json.loads(content)
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

    gen_key: str = ""   # set by each subclass ("A", "B", etc.)
    name: str = ""
    num_batches: int = 10  # legacy default for metadata generators

    @property
    def items_per_chunk(self):
        return ITEMS_PER_CALL

    @property
    def SUPPORTED_FORMATS(self):
        return GENERATOR_SPEC[self.gen_key]["formats"]

    @property
    def needs_corpus(self):
        return GENERATOR_SPEC[self.gen_key]["corpus"]

    @property
    def model(self):
        """Model override for this generator, or the global MODEL default."""
        return GENERATOR_MODEL_OVERRIDES.get(self.gen_key, MODEL)

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

    def expand_chunk_to_tasks(self, chunk, period, start_year, end_year):
        """Expand a chunk into one or more (format_key, prompt) pairs.

        Default: one task per chunk using build_prompt(). format_key=None means
        "render the response into all SUPPORTED_FORMATS".

        Override in generators like Gen G where each chunk needs SEPARATE API
        calls per format (plain-text prompts, no JSON bundling). Return a list
        of (format_key, prompt_text) pairs; when format_key is set, the
        response is rendered ONLY into that format.
        """
        return [(None, self.build_prompt(chunk, period, start_year, end_year))]

    def is_plaintext_format(self, fmt):
        """Whether responses for this format are plain text (not JSON).

        Override in generators that need plain-text parsing (e.g. Gen G rephrase).
        """
        return False

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
        print(f"Model: {self.model}")

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

        # Build task list. Each chunk may expand into multiple tasks (e.g. Gen G
        # produces 4 tasks per chunk, one per rephrase format).
        tasks = []
        for doc_name, text in docs:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for i, chunk in enumerate(chunks):
                for fmt_key, prompt_text in self.expand_chunk_to_tasks(
                    chunk, period, start_year, end_year
                ):
                    tasks.append((client, doc_name, i, chunk, prompt_text, fmt_key,
                                  period, start_year, end_year))

        # Cap tasks to match target. Each task produces:
        #   - (ITEMS_PER_CALL items) × (num_formats convs) if fmt_key=None (default)
        #   - exactly 1 conversation if fmt_key is set (Gen G: one format per task)
        if target_examples:
            from src.post_training.config import ITEMS_PER_CALL
            if any(t[5] is not None for t in tasks):  # any task has a fmt_key → Gen G style
                max_calls = target_examples  # one conversation per task
            else:
                max_calls = target_examples // (ITEMS_PER_CALL * len(self.SUPPORTED_FORMATS))
            if len(tasks) > max_calls:
                print(f"Capping tasks: {len(tasks):,} -> {max_calls:,} (target: {target_examples:,})")
                tasks = tasks[:max_calls]

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
        """Process a single task. Returns {fmt: [conversations]}.

        Task tuple: (client, doc_name, chunk_idx, chunk, prompt, fmt_key,
                     period, start_year, end_year)

        When fmt_key is None (default): render response into ALL SUPPORTED_FORMATS.
        When fmt_key is set (Gen G): render response only into that specific format.

        Each conversation is wrapped with metadata for document-level splitting:
        {"messages": [...], "doc_name": ..., "chunk_idx": ..., "generator": ..., "format": ...}
        """
        client, doc_name, chunk_idx, chunk, prompt, fmt_key, period, start_year, end_year = task

        plaintext = fmt_key is not None and self.is_plaintext_format(fmt_key)
        response = call_api(client, prompt, model=self.model, plaintext=plaintext)

        if response is None:
            return {}

        items = self.parse_response(response)
        results = {}
        # If fmt_key is set, only render into that one format. Else iterate all formats.
        target_formats = (fmt_key,) if fmt_key is not None else self.SUPPORTED_FORMATS
        for item in items:
            for fmt in target_formats:
                conv = self.format_conversation(item, fmt, source_chunk=chunk)
                if conv is None:
                    continue
                valid, err = validate_conversation(conv)
                if valid:
                    wrapped = {
                        "messages": conv,
                        "doc_name": doc_name,
                        "chunk_idx": chunk_idx,
                        "generator": self.name,
                        "format": fmt,
                    }
                    results.setdefault(fmt, []).append(wrapped)
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

        # Build task list. Each chunk may expand into multiple tasks (e.g. Gen G
        # produces one task per rephrase format). Task tuple shape:
        #   (custom_id, prompt, chunk_text, doc_name, chunk_idx, fmt_key)
        if self.needs_corpus:
            docs = self._load_documents(paths, collections, max_docs)
            if not docs:
                print("No documents found!")
                return None
            tasks = []
            for doc_idx, (doc_name, text) in enumerate(docs):
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for chunk_idx, chk in enumerate(chunks):
                    for task_num, (fmt_key, prompt) in enumerate(
                        self.expand_chunk_to_tasks(chk, period, start_year, end_year)
                    ):
                        # Encode task_num (format slot) into custom_id when fmt_key is set,
                        # so each format's response gets its own custom_id.
                        if fmt_key is None:
                            cid = f"{self.name}_{doc_idx}_{chunk_idx}"
                        else:
                            cid = f"{self.name}_{doc_idx}_{chunk_idx}_{fmt_key}"
                        tasks.append((cid, prompt, chk, doc_name, chunk_idx, fmt_key))
        else:
            tasks = self._build_metadata_tasks(
                period, start_year, end_year, target_examples,
            )

        if not tasks:
            print("No tasks to submit!")
            return None

        # Cap tasks to match target
        if target_examples and self.needs_corpus:
            from src.post_training.config import ITEMS_PER_CALL
            any_fmt_key = len(tasks[0]) >= 6 and tasks[0][5] is not None
            if any_fmt_key:
                max_calls = target_examples  # one conversation per task
            else:
                max_calls = target_examples // (ITEMS_PER_CALL * len(self.SUPPORTED_FORMATS))
            if len(tasks) > max_calls:
                print(f"Capping tasks: {len(tasks):,} -> {max_calls:,} (target: {target_examples:,})")
                tasks = tasks[:max_calls]

        total = len(tasks)
        print(f"Submitting {total:,} batch requests")

        # Write manifest (custom_id -> chunk/doc metadata + optional fmt_key)
        manifest_path = batch_dir / f"{self.name}_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for task_tuple in tasks:
                cid = task_tuple[0]
                chunk_val = task_tuple[2]
                entry = {"custom_id": cid}
                if chunk_val is not None:
                    entry["chunk_text"] = chunk_val
                # Corpus tasks: (cid, prompt, chunk, doc_name, chunk_idx, fmt_key)
                if len(task_tuple) >= 5:
                    entry["doc_name"] = task_tuple[3]
                    entry["chunk_idx"] = task_tuple[4]
                if len(task_tuple) >= 6 and task_tuple[5] is not None:
                    entry["fmt_key"] = task_tuple[5]
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Build request dicts. Gen G requests use plaintext (no response_format)
        # since the rephrase prompts return raw text. Everything else uses JSON.
        requests = []
        for task_tuple in tasks:
            cid, prompt = task_tuple[0], task_tuple[1]
            fmt_key = task_tuple[5] if len(task_tuple) >= 6 else None
            plaintext = fmt_key is not None and self.is_plaintext_format(fmt_key)
            req = {
                "custom_id": cid,
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.7,
            }
            if not plaintext:
                req["response_format"] = {"type": "json_object"}
            requests.append(req)

        # Split by serialized byte size (OpenAI gpt-4o-mini limit: 200MB per file)
        # Target 180MB per chunk to leave safety margin.
        MAX_BYTES = 180 * 1024 * 1024
        chunks = []
        current, current_bytes = [], 0
        for req in requests:
            # Approximate size of one serialized line (request body + JSON overhead).
            # Matches format in create_batch_request_file.
            line_bytes = len(json.dumps({
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req.get("model", MODEL),
                    "messages": req["messages"],
                    "max_tokens": req.get("max_tokens", 256),
                    "response_format": {"type": "json_object"},
                    **({"temperature": req["temperature"]} if "temperature" in req else {}),
                },
            })) + 1  # newline
            if current and current_bytes + line_bytes > MAX_BYTES:
                chunks.append(current)
                current, current_bytes = [], 0
            current.append(req)
            current_bytes += line_bytes
        if current:
            chunks.append(current)

        batch_ids = []
        for ci, chunk in enumerate(chunks):
            suffix = f"_part{ci}" if len(chunks) > 1 else ""
            request_path = batch_dir / f"{self.name}_requests{suffix}.jsonl"
            create_batch_request_file(chunk, str(request_path))

            batch_id = submit_batch(
                str(request_path),
                description=f"{self.name}_{period}{suffix}",
            )
            batch_ids.append(batch_id)
            print(f"  Part {ci+1}/{len(chunks)}: {batch_id} ({len(chunk):,} requests)")

        # Save batch IDs (one per line)
        id_path = batch_dir / f"{self.name}_batch_id.txt"
        with open(id_path, "w") as f:
            f.write("\n".join(batch_ids))
        print(f"  {len(batch_ids)} batch(es) saved: {id_path.name}")

        return batch_ids[0] if len(batch_ids) == 1 else batch_ids

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

        # Read batch ID(s) — one per line
        id_path = batch_dir / f"{self.name}_batch_id.txt"
        if not id_path.exists():
            print(f"No batch ID found at {id_path}")
            return None
        batch_ids = [line.strip() for line in id_path.read_text().strip().splitlines() if line.strip()]

        # Download results from all batches
        results = []
        for bi, batch_id in enumerate(batch_ids):
            suffix = f"_part{bi}" if len(batch_ids) > 1 else ""
            results_path = batch_dir / f"{self.name}_results{suffix}.jsonl"
            part_results = download_batch_results(batch_id, str(results_path))
            if part_results is None:
                print(f"Batch {bi+1}/{len(batch_ids)} ({batch_id}) not complete yet")
                return None
            results.extend(part_results)
            print(f"  Part {bi+1}/{len(batch_ids)}: {len(part_results):,} results")

        # Load manifest (custom_id -> {chunk_text, doc_name, chunk_idx})
        manifest = {}
        manifest_path = batch_dir / f"{self.name}_manifest.jsonl"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    manifest[entry["custom_id"]] = {
                        "chunk_text": entry.get("chunk_text"),
                        "doc_name": entry.get("doc_name", "unknown"),
                        "chunk_idx": entry.get("chunk_idx", 0),
                        "fmt_key": entry.get("fmt_key"),  # Gen G: one format per task
                    }

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
            meta = manifest.get(custom_id, {})
            fmt_key = meta.get("fmt_key") if isinstance(meta, dict) else None

            # Plain-text responses (Gen G rephrase): don't try to parse as JSON;
            # the response body IS the text payload. Wrap it as a single item.
            if fmt_key is not None and self.is_plaintext_format(fmt_key):
                # response_body is the raw assistant text from the batch result
                items = [{"text": response_body.strip() if isinstance(response_body, str) else ""}]
            else:
                try:
                    response = json.loads(response_body)
                except json.JSONDecodeError:
                    parsed_fail += 1
                    continue
                items = self.parse_response(response)

            chunk_val = meta.get("chunk_text") if isinstance(meta, dict) else meta

            # If fmt_key is set (Gen G), render only into that format.
            target_formats = (fmt_key,) if fmt_key is not None else self.SUPPORTED_FORMATS

            for item in items:
                if not isinstance(item, dict):
                    continue
                for fmt in target_formats:
                    conv = self.format_conversation(item, fmt, source_chunk=chunk_val)
                    if conv is None:
                        continue
                    valid, _ = validate_conversation(conv)
                    if valid:
                        wrapped = {
                            "messages": conv,
                            "doc_name": meta.get("doc_name", "unknown") if isinstance(meta, dict) else "unknown",
                            "chunk_idx": meta.get("chunk_idx", 0) if isinstance(meta, dict) else 0,
                            "generator": self.name,
                            "format": fmt,
                        }
                        all_results[fmt].append(wrapped)
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
                            clean_ocr_text(row["text"]),
                        ))
            elif collection_dir.exists():
                for txt_file in sorted(collection_dir.glob("*.txt")):
                    text = txt_file.read_text(encoding="utf-8")
                    if text.strip():
                        docs.append((f"{collection}/{txt_file.stem}", clean_ocr_text(text)))

        if max_docs and len(docs) > max_docs:
            import random
            rng = random.Random(42)
            docs = rng.sample(docs, min(max_docs, len(docs)))

        print(f"Loaded {len(docs)} documents from {len(collections)} collection(s)")
        return docs
