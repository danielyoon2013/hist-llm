"""
Three-level deduplication for synthetic data.

Level 1: Exact hash (SHA-256 of normalized user message)
Level 2: Near-duplicate (MinHash + LSH, threshold 0.8)
Level 3: Cross-generator (exact hash across different base generators)

Format variants of the same generator (e.g. gen_b_cot_mc4 vs gen_b_cot_open)
are NOT considered duplicates — they have the same question but different
answer formats, which is intentional.

Requires: pip install datasketch
"""

import os
import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict

from src.post_training.utils import read_jsonl, write_jsonl


# Format suffixes appended by multi-format generators.
# Used to extract base generator name from filenames like gen_b_cot_mc4.
_FORMAT_SUFFIXES = ("_mc4_passage", "_mc2_passage", "_mc4", "_mc2", "_open", "_cot")


def _base_generator_name(filename_stem):
    """Extract base generator name by stripping format suffix.

    gen_b_cot_mc4     -> gen_b_cot
    gen_a_factual_open -> gen_a_factual
    gen_c_comprehension_mc4_passage -> gen_c_comprehension
    some_other_file   -> some_other_file  (no suffix match)
    """
    for suffix in _FORMAT_SUFFIXES:
        if filename_stem.endswith(suffix):
            return filename_stem[:-len(suffix)]
    return filename_stem


def _normalize(text):
    """Normalize text for comparison: lowercase, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def _hash_text(text):
    """SHA-256 hash of normalized text."""
    return hashlib.sha256(_normalize(text).encode('utf-8')).hexdigest()


def _get_user_message(conv):
    """Extract the first user message from a conversation."""
    for msg in conv:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def _shingles(text, n=5):
    """Character n-gram shingles for MinHash."""
    text = _normalize(text)
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


# ---------------------------------------------------------------------------
# Level 1: Exact hash dedup
# ---------------------------------------------------------------------------

def dedup_exact(conversations):
    """Remove exact duplicates by SHA-256 of user message.

    Returns (deduped_list, num_removed).
    """
    seen = set()
    deduped = []
    removed = 0

    for conv in conversations:
        h = _hash_text(_get_user_message(conv))
        if h in seen:
            removed += 1
        else:
            seen.add(h)
            deduped.append(conv)

    return deduped, removed


# ---------------------------------------------------------------------------
# Level 2: Near-duplicate (MinHash + LSH)
# ---------------------------------------------------------------------------

def dedup_minhash(conversations, threshold=0.8, num_perm=128):
    """Remove near-duplicates using MinHash + LSH.

    Args:
        conversations: List of conversations
        threshold: Jaccard similarity threshold (default 0.8)
        num_perm: Number of MinHash permutations (default 128)

    Returns (deduped_list, num_removed).
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  [WARN] datasketch not installed, skipping MinHash dedup")
        print("  Install with: pip install datasketch")
        return conversations, 0

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    removed_indices = set()

    # Build MinHash for each conversation
    for i, conv in enumerate(conversations):
        text = _get_user_message(conv)
        shings = _shingles(text)

        mh = MinHash(num_perm=num_perm)
        for s in shings:
            mh.update(s.encode('utf-8'))
        minhashes.append(mh)

        # Check for near-duplicates
        key = f"doc_{i}"
        result = lsh.query(mh)
        if result:
            removed_indices.add(i)
        else:
            try:
                lsh.insert(key, mh)
            except ValueError:
                # Duplicate key — already inserted
                removed_indices.add(i)

    deduped = [conv for i, conv in enumerate(conversations)
               if i not in removed_indices]

    return deduped, len(removed_indices)


# ---------------------------------------------------------------------------
# Level 3: Cross-generator exact dedup
# ---------------------------------------------------------------------------

def dedup_cross_generator(generator_files, priority_order=None):
    """Remove exact duplicates across different base generators.

    Format variants of the same generator are NOT considered duplicates.
    For example, gen_b_cot_mc4 and gen_b_cot_open share the same questions
    by design — only gen_a_factual_open vs gen_f_instruct_open would dedup.

    Higher-priority generators keep their examples.

    Args:
        generator_files: dict mapping generator_name -> list of conversations
        priority_order: list of base generator names in priority order
                       (default: A > B > C > D > E > F > G > H)

    Returns dict mapping generator_name -> deduped conversations, plus stats.
    """
    if priority_order is None:
        priority_prefixes = [
            "gen_a_", "gen_b_", "gen_c_", "gen_d_",
            "gen_e_", "gen_f_", "gen_g_", "gen_h_",
        ]
        all_names = sorted(generator_files.keys())
        priority_order = []
        for prefix in priority_prefixes:
            for name in all_names:
                if name.startswith(prefix) and name not in priority_order:
                    priority_order.append(name)
        for name in all_names:
            if name not in priority_order:
                priority_order.append(name)

    # Track seen hashes per base generator so format variants don't collide.
    # The global set only tracks hashes from OTHER base generators.
    # Key insight: hash is added to global_seen under the base generator name
    # that first produced it. Format siblings share the same base name so
    # they won't collide with each other.
    base_seen = defaultdict(set)   # base_name -> set of hashes seen
    global_seen = set()            # union of all base_seen values

    result = {}
    stats = {}

    for gen_name in priority_order:
        if gen_name not in generator_files:
            continue

        base_name = _base_generator_name(gen_name)
        conversations = generator_files[gen_name]
        deduped = []
        removed = 0

        for conv in conversations:
            h = _hash_text(_get_user_message(conv))
            # Skip if another base generator already claimed this hash
            if h in global_seen and h not in base_seen[base_name]:
                removed += 1
            else:
                base_seen[base_name].add(h)
                global_seen.add(h)
                deduped.append(conv)

        result[gen_name] = deduped
        stats[gen_name] = {"before": len(conversations), "after": len(deduped),
                           "removed": removed}

    # Include any generators not in priority_order
    for gen_name, conversations in generator_files.items():
        if gen_name not in result:
            base_name = _base_generator_name(gen_name)
            deduped = []
            removed = 0
            for conv in conversations:
                h = _hash_text(_get_user_message(conv))
                if h in global_seen and h not in base_seen[base_name]:
                    removed += 1
                else:
                    base_seen[base_name].add(h)
                    global_seen.add(h)
                    deduped.append(conv)
            result[gen_name] = deduped
            stats[gen_name] = {"before": len(conversations), "after": len(deduped),
                               "removed": removed}

    return result, stats


# ---------------------------------------------------------------------------
# Full dedup pipeline for a directory
# ---------------------------------------------------------------------------

def dedup_directory(input_dir, output_dir):
    """Run 3-level dedup on all generator JSONL files.

    Args:
        input_dir: Directory with gen_*.jsonl files
        output_dir: Directory to write deduped files

    Returns stats dict.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_stats = {}

    # Level 1 + 2: per-generator
    generator_data = {}
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        gen_name = jsonl_file.stem
        conversations = read_jsonl(str(jsonl_file))
        original = len(conversations)

        # Level 1: exact
        conversations, exact_removed = dedup_exact(conversations)

        # Level 2: minhash
        conversations, minhash_removed = dedup_minhash(conversations)

        generator_data[gen_name] = conversations
        all_stats[gen_name] = {
            "original": original,
            "after_exact": original - exact_removed,
            "exact_removed": exact_removed,
            "after_minhash": len(conversations),
            "minhash_removed": minhash_removed,
        }

        print(f"  {gen_name}: {original} -> {original - exact_removed} (exact) "
              f"-> {len(conversations)} (minhash)")

    # Level 3: cross-generator
    print("  Cross-generator dedup...")
    generator_data, cross_stats = dedup_cross_generator(generator_data)

    for gen_name, data in generator_data.items():
        output_path = output_dir / f"{gen_name}.jsonl"
        write_jsonl(data, str(output_path), validate=False)

        cs = cross_stats.get(gen_name, {})
        all_stats[gen_name]["cross_removed"] = cs.get("removed", 0)
        all_stats[gen_name]["final"] = len(data)

        print(f"  {gen_name}: -> {len(data)} (cross-gen, removed {cs.get('removed', 0)})")

    return all_stats
