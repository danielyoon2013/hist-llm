"""
Format and content validation for generated synthetic data.

Reuses validate_conversation() from utils.py and adds content-level checks
(min/max length, empty content).
"""

import os
import json
from pathlib import Path

from src.post_training.utils import validate_conversation, read_jsonl, write_jsonl


# Content thresholds
MIN_CONTENT_LENGTH = 10   # reject very short answers
MAX_CONTENT_LENGTH = 10000  # reject suspiciously long content


def validate_content(messages):
    """Additional content-level validation beyond format checks.

    Returns (is_valid, error_message).
    MC-format responses (single letter A-D) are exempt from min length.
    """
    for i, msg in enumerate(messages):
        content = msg["content"]
        # Skip min-length check for assistant MC responses (single letter A-D)
        is_mc_response = (msg["role"] == "assistant" and content.strip() in ("A", "B", "C", "D"))
        if not is_mc_response and len(content.strip()) < MIN_CONTENT_LENGTH:
            return False, f"Message {i} content too short ({len(content)} chars)"
        if len(content) > MAX_CONTENT_LENGTH:
            return False, f"Message {i} content too long ({len(content)} chars)"
    return True, ""


def validate_file(input_path, output_path):
    """Validate all conversations in a JSONL file.

    Args:
        input_path: Path to input JSONL
        output_path: Path to write validated output

    Returns:
        dict with counts: total, valid, invalid_format, invalid_content
    """
    conversations = read_jsonl(str(input_path))

    valid = []
    stats = {"total": len(conversations), "valid": 0,
             "invalid_format": 0, "invalid_content": 0}

    for conv in conversations:
        # Format check (from utils.py)
        ok, err = validate_conversation(conv)
        if not ok:
            stats["invalid_format"] += 1
            continue

        # Content check
        ok, err = validate_content(conv)
        if not ok:
            stats["invalid_content"] += 1
            continue

        valid.append(conv)
        stats["valid"] += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_jsonl(valid, str(output_path), validate=False)  # already validated

    return stats


def validate_directory(input_dir, output_dir):
    """Validate all JSONL files in a directory.

    Args:
        input_dir: Directory containing gen_*.jsonl files
        output_dir: Directory to write validated files

    Returns:
        dict mapping filename -> stats
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_stats = {}

    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        output_path = output_dir / jsonl_file.name
        stats = validate_file(jsonl_file, output_path)
        all_stats[jsonl_file.name] = stats

        pct = 100 * stats["valid"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {jsonl_file.name}: {stats['valid']}/{stats['total']} valid "
              f"({pct:.1f}%) | format_err={stats['invalid_format']} "
              f"content_err={stats['invalid_content']}")

    return all_stats
