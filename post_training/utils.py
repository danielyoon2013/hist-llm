"""
Shared utilities for post-training data generation.

Three components:
1. OpenAI client — API calls with retry, structured output, batch API
2. JSONL I/O — read/write/validate in nanochat's CustomJSON format
3. Parquet reader — sample documents from base_data shards
"""

import os
import json
import time
import random
import glob
from pathlib import Path
from openai import OpenAI

from src.post_training.config import load_api_key, MODEL, MAX_RETRIES, RETRY_BASE_DELAY


# ===========================================================================
# 1. OpenAI Client
# ===========================================================================

_client = None

def get_client():
    """Lazy-initialize the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=load_api_key())
    return _client


def call_openai(messages, response_format=None, model=None, max_tokens=2048):
    """
    Call OpenAI chat completion with retry and exponential backoff.

    Args:
        messages: List of message dicts (role, content)
        response_format: Optional structured output format (JSON schema)
        model: Model name (default: config.MODEL)
        max_tokens: Max output tokens

    Returns:
        The assistant's response content as a string
    """
    client = get_client()
    model = model or MODEL

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_DELAY ** (attempt + 1)
                print(f"  API error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def call_openai_json(messages, model=None, max_tokens=2048):
    """Call OpenAI and parse the response as JSON."""
    text = call_openai(
        messages,
        response_format={"type": "json_object"},
        model=model,
        max_tokens=max_tokens,
    )
    return json.loads(text)


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

def create_batch_request_file(requests, output_path):
    """
    Write a JSONL file of batch API requests.

    Args:
        requests: List of dicts, each with:
            - custom_id: unique string ID
            - messages: list of message dicts
            - model: model name (optional, uses default)
            - max_tokens: int (optional, default 256)
        output_path: Path to write the JSONL file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for req in requests:
            body = {
                "model": req.get("model", MODEL),
                "messages": req["messages"],
                "max_tokens": req.get("max_tokens", 256),
            }
            # Only attach response_format when the request asks for it.
            # Gen G rephrase prompts are plain-text and do not include "json" in
            # their message, which would trigger an OpenAI validation error
            # ("'messages' must contain the word 'json' in some form").
            if "response_format" in req:
                body["response_format"] = req["response_format"]
            if "temperature" in req:
                body["temperature"] = req["temperature"]
            line = {
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line) + "\n")
    print(f"Wrote {len(requests)} batch requests to {output_path}")


def submit_batch(request_file_path, description="temporal_filter"):
    """Submit a batch job to OpenAI and return the batch ID."""
    client = get_client()

    # Upload the request file
    with open(request_file_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"Uploaded file: {uploaded.id}")

    # Create the batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"Batch created: {batch.id} (status: {batch.status})")
    return batch.id


def check_batch_status(batch_id):
    """Check the status of a batch job. Returns the batch object."""
    client = get_client()
    batch = client.batches.retrieve(batch_id)
    print(f"Batch {batch_id}: status={batch.status}, "
          f"completed={batch.request_counts.completed}/{batch.request_counts.total}, "
          f"failed={batch.request_counts.failed}")
    return batch


def download_batch_results(batch_id, output_path):
    """Download batch results to a JSONL file. Returns list of (custom_id, response_body) tuples."""
    client = get_client()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"Batch not complete yet (status: {batch.status})")
        return None

    # Download the output file
    content = client.files.content(batch.output_file_id)
    with open(output_path, "wb") as f:
        f.write(content.read())

    # Parse results
    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            custom_id = data["custom_id"]
            response_body = data["response"]["body"]["choices"][0]["message"]["content"]
            results.append((custom_id, response_body))

    print(f"Downloaded {len(results)} results to {output_path}")
    return results


# ===========================================================================
# 2. JSONL I/O (nanochat CustomJSON format)
# ===========================================================================

def validate_conversation(messages):
    """
    Validate a conversation against nanochat's CustomJSON format.
    Returns (is_valid, error_message).

    Accepts either:
    - A bare list of message dicts (legacy format)
    - A dict with "messages" key (metadata-wrapped format)

    Rules (from nanochat/tasks/customjson.py lines 42-50):
    - Must be a list
    - At least 2 messages
    - Alternating user/assistant roles starting with user
    - All content must be strings
    """
    # Unwrap metadata dict if present
    if isinstance(messages, dict):
        messages = messages.get("messages", [])
    if not isinstance(messages, list):
        return False, f"Expected list, got {type(messages)}"
    if len(messages) < 2:
        return False, f"Need at least 2 messages, got {len(messages)}"
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Message {i} is not a dict"
        if "role" not in msg or "content" not in msg:
            return False, f"Message {i} missing 'role' or 'content'"
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected_role:
            return False, f"Message {i} has role '{msg['role']}', expected '{expected_role}'"
        if not isinstance(msg["content"], str):
            return False, f"Message {i} content must be a string, got {type(msg['content'])}"
    return True, ""


def write_jsonl(conversations, output_path, validate=True):
    """
    Write conversations to a JSONL file in CustomJSON format.

    Args:
        conversations: List of conversations, each is a list of {role, content} dicts
        output_path: Path to write
        validate: If True, validate each conversation before writing
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            if validate:
                valid, err = validate_conversation(conv)
                if not valid:
                    skipped += 1
                    continue
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} conversations to {output_path}" +
          (f" (skipped {skipped} invalid)" if skipped > 0 else ""))


def append_jsonl(conversations, output_path, validate=True):
    """Append conversations to an existing JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for conv in conversations:
            if validate:
                valid, err = validate_conversation(conv)
                if not valid:
                    continue
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            written += 1
    return written


def read_jsonl(path):
    """Read all conversations from a JSONL file."""
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    return conversations


def count_jsonl(path):
    """Count lines in a JSONL file without loading everything into memory."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ===========================================================================
# 3. Parquet Reader
# ===========================================================================

def list_shards(base_data_dir):
    """List all parquet shard files sorted by name."""
    pattern = os.path.join(str(base_data_dir), "shard_*.parquet")
    files = sorted(glob.glob(pattern))
    return files


def sample_documents(base_data_dir, num_docs, seed=42):
    """
    Sample random documents from parquet shards.

    Args:
        base_data_dir: Path to base_data/ directory containing shard_*.parquet files
        num_docs: Total number of documents to sample
        seed: Random seed for reproducibility

    Returns:
        List of document text strings
    """
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    shards = list_shards(base_data_dir)
    assert len(shards) > 0, f"No parquet shards found in {base_data_dir}"

    # Calculate how many docs per shard (spread evenly)
    docs_per_shard = max(1, num_docs // len(shards))
    remaining = num_docs - docs_per_shard * len(shards)

    documents = []
    for shard_path in shards:
        pf = pq.ParquetFile(shard_path)
        # Read all row groups and collect texts
        all_texts = []
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            all_texts.extend(rg.column("text").to_pylist())

        # Sample from this shard
        n = docs_per_shard + (1 if remaining > 0 else 0)
        if remaining > 0:
            remaining -= 1
        n = min(n, len(all_texts))
        sampled = rng.sample(all_texts, n)
        documents.extend(sampled)

        if len(documents) >= num_docs:
            break

    return documents[:num_docs]
