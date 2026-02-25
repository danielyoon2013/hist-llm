"""
Base class for all synthetic data generators.

Extracts common patterns from run_direct.py:
- Text chunking (6000 chars, 300 overlap)
- OpenAI API calls with retry (ThreadPoolExecutor)
- JSONL output in nanochat CustomJSON format
"""

import os
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.post_training.config import PERIODS, get_paths, load_api_key, MODEL, MAX_RETRIES, RETRY_BASE_DELAY
from src.post_training.utils import validate_conversation, write_jsonl


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


class BaseGenerator(ABC):
    """Abstract base class for synthetic data generators."""

    name: str = ""
    items_per_chunk: int = 3
    needs_corpus: bool = True  # False for D (temporal) and H (anti-halluc)

    @abstractmethod
    def build_prompt(self, chunk, period, start_year, end_year):
        """Build the generation prompt for a text chunk (or metadata)."""
        ...

    @abstractmethod
    def parse_response(self, response):
        """Parse API response dict into list of raw items."""
        ...

    @abstractmethod
    def format_conversation(self, item):
        """Convert a raw item into nanochat conversation format.
        Returns list of {role, content} dicts."""
        ...

    def run(self, period, collections=None, max_workers=50, max_docs=None,
            chunk_size=6000, overlap=300):
        """Run this generator for a period.

        Args:
            period: Period key (e.g. "1900_1949")
            collections: Collection names to process. None = all available.
            max_workers: Concurrent API calls
            max_docs: Limit documents per collection (for testing)
            chunk_size: Characters per chunk
            overlap: Overlap between chunks

        Returns:
            Path to output JSONL file
        """
        from openai import OpenAI

        paths = get_paths(period)
        start_year, end_year = PERIODS[period]

        generators_dir = paths["synthetic_dir"] / "by_generator"
        os.makedirs(generators_dir, exist_ok=True)
        output_path = generators_dir / f"{self.name}.jsonl"

        api_key = load_api_key()
        client = OpenAI(api_key=api_key)

        print(f"Generator: {self.name}")
        print(f"Period: {period} ({start_year}-{end_year})")
        print(f"Max workers: {max_workers}")
        print(f"Model: {MODEL}")

        if not self.needs_corpus:
            return self._run_metadata_based(
                client, period, start_year, end_year, max_workers, output_path
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
        print(f"Output: {output_path}")

        # Process with ThreadPoolExecutor
        conversations = []
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
                    convs = future.result()
                    conversations.extend(convs)
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)

                done = completed + failed
                if done % 25 == 0 or done == total_tasks:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (total_tasks - done) / rate if rate > 0 else 0
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"{done:,}/{total_tasks:,} ({100*done/total_tasks:.1f}%) "
                          f"Rate: {rate:.1f}/s | OK: {completed} Failed: {failed}",
                          flush=True)

        elapsed = time.time() - start_time
        write_jsonl(conversations, str(output_path))
        print(f"Complete in {elapsed:.1f}s. "
              f"{len(conversations):,} conversations. Failed: {failed}")
        return output_path

    def _process_task(self, task):
        """Process a single chunk. Runs in thread pool."""
        client, doc_name, chunk_idx, chunk, period, start_year, end_year = task

        prompt = self.build_prompt(chunk, period, start_year, end_year)
        response = call_api(client, prompt)

        if response is None:
            return []

        items = self.parse_response(response)
        conversations = []
        for item in items:
            conv = self.format_conversation(item)
            valid, err = validate_conversation(conv)
            if valid:
                conversations.append(conv)
        return conversations

    def _run_metadata_based(self, client, period, start_year, end_year,
                            max_workers, output_path):
        """Override for generators that don't need corpus (D, H)."""
        raise NotImplementedError(
            f"{self.name} has needs_corpus=False but doesn't implement "
            f"_run_metadata_based()"
        )

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
