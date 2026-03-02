"""
Central configuration for post-training data generation.
All scripts import period definitions, paths, and API settings from here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root and API key
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # hist_LLM/
DATA_ROOT = Path("D:/hist_LLM")

def load_api_key():
    key_path = PROJECT_ROOT / "key.txt"
    with open(key_path, "r") as f:
        return f.read().strip()

# ---------------------------------------------------------------------------
# Period definitions
# ---------------------------------------------------------------------------

PERIODS = {
    "1678_1849": (1678, 1849),
    "1850_1899": (1850, 1899),
    "1900_1949": (1900, 1949),
    "1950_1999": (1950, 1999),
    "2000_2009": (2000, 2009),
    "2010_2023": (2010, 2023),
}

# ---------------------------------------------------------------------------
# Path resolution — all paths derived from period name
# ---------------------------------------------------------------------------

def get_paths(period: str):
    """Return a dict of all relevant paths for a given period."""
    assert period in PERIODS, f"Unknown period '{period}'. Choose from: {list(PERIODS.keys())}"
    start_year, end_year = PERIODS[period]

    # Input data paths (D:/hist_LLM/periods/{period}/)
    period_data = DATA_ROOT / "periods" / period

    # Model data paths (checkpoints, tokenizer, evals)
    model_data = period_data / "model"

    return {
        # Period info
        "period": period,
        "start_year": start_year,
        "end_year": end_year,

        # Input data paths
        "base_data_dir": period_data / "base_data",
        "posttraining_dir": period_data / "posttraining_data",
        "final_dir": period_data / "posttraining_data" / "final",
        "final_filtered_dir": period_data / "posttraining_data" / "final" / "filtered",
        "final_removed_dir": period_data / "posttraining_data" / "final" / "removed",
        "final_train_dir": period_data / "posttraining_data" / "final" / "train",
        "final_test_dir": period_data / "posttraining_data" / "final" / "test",
        "identity_output": period_data / "posttraining_data" / "identity.jsonl",
        "identity_nanochat": period_data / "identity_conversations.jsonl",
        "corpus_qa_output": period_data / "posttraining_data" / "synthetic" / "output" / "corpus_qa.jsonl",  # legacy
        "corpus_qa_v2_output": period_data / "posttraining_data" / "synthetic" / "output" / "corpus_qa_v2.jsonl",  # legacy
        "hist_corpus_qa_output": period_data / "posttraining_data" / f"hist_corpus_qa_{period}.jsonl",
        "LAB_scores_dir": period_data / "posttraining_data" / "LAB_scores",
        "lab_eval_dir": period_data / "posttraining_data" / "eval",
        "synthetic_dir": period_data / "posttraining_data" / "synthetic",
        "generators_dir": period_data / "posttraining_data" / "synthetic" / "by_generator",
        "batch_temp_dir": period_data / "posttraining_data" / "synthetic" / "batch_temp",
        "quality_dir": period_data / "posttraining_data" / "quality",
        "validated_dir": period_data / "posttraining_data" / "quality" / "validated",
        "deduped_dir": period_data / "posttraining_data" / "quality" / "deduped",
        "metadata_index": period_data / "posttraining_data" / "synthetic" / "document_metadata.parquet",

        # Shared instruct datasets (not period-specific)
        "instruct_data_dir": DATA_ROOT / "instruct_data",

        # Model data paths
        "model_data_dir": model_data,
        "tokenizer_dir": model_data / "tokenizer",
        "base_checkpoints": model_data / "base_checkpoints",
        "mid_checkpoints": model_data / "mid_checkpoints",
        "chatsft_checkpoints": model_data / "chatsft_checkpoints",
        "eval_dir": model_data / "base_eval",
        "report_dir": model_data / "report",

        # Main historical corpus (D:/hist_LLM/corpus/)
        "raw_data_root": DATA_ROOT / "corpus" / "raw",
        "classified_root": DATA_ROOT / "corpus" / "classified",

        # Additional data sources (D:/hist_LLM/additional_data/raw/)
        "additional_data_dir": DATA_ROOT / "additional_data" / "raw",
        "nyt_filtered_dir": DATA_ROOT / "additional_data" / "raw" / "news_archives" / "NYT_filtered_500char",
        "economist_dir": DATA_ROOT / "additional_data" / "raw" / "news_archives" / "Economist",
        "ft_dir": DATA_ROOT / "additional_data" / "raw" / "news_archives" / "FT",
        "newswire_dir": DATA_ROOT / "additional_data" / "raw" / "newswire",
    }

# ---------------------------------------------------------------------------
# Generation targets (per period)
# ---------------------------------------------------------------------------

DEFAULT_TARGET = 1_000_000      # 1M mid-train examples per period
DEFAULT_SFT_SIZE = 10_000       # 10K SFT examples (1% proportional subsample)
DEFAULT_TEST_RATIO = 0.05       # 5% holdout for training-loss monitoring

# ---------------------------------------------------------------------------
# Generator specification — single source of truth
# ---------------------------------------------------------------------------
# Allocation is purely determined by format count: each format slot gets
# an equal share of the target. No manual percentages to maintain.
#
# At 1M target with 9 total format slots:
#   per_slot = 111,111  →  A(2)=222,222  B(2)=222,222  ...  F(1)=111,111
#
# All generators are corpus-based (need document text).

ITEMS_PER_CALL = 2      # items requested per API call (all generators)
CHUNKS_PER_DOC = 2      # average chunks per document (6000 chars, 300 overlap)

GENERATOR_SPEC = {
    "A": {"formats": ("mc4", "open"),        "corpus": True},
    "B": {"formats": ("mc4", "cot"),         "corpus": True},   # removed "open" (redundant — cot already contains the answer)
    "C": {"formats": ("mc4_passage",),       "corpus": True},   # removed "mc2_passage" (artificial 2-choice reduction)
    "D": {"formats": ("open", "cot"),        "corpus": True},
    "E": {"formats": ("mc4",),              "corpus": True},   # removed "mc2" (discards 2 distractors from mc4)
    "F": {"formats": ("mc4_passage",),       "corpus": True},
}


def compute_plan(target=DEFAULT_TARGET, gen_keys=None):
    """Derive per-generator targets and doc counts from format counts.

    Returns:
        {
            "target": 1_000_000,
            "generators": {
                "A": {"target": 222,222, "per_format": 111,111, "docs_needed": 27,778},
                "B": {"target": 222,222, "per_format": 111,111, "docs_needed": 27,778},
                ...
            }
        }
    """
    if gen_keys is None:
        gen_keys = list(GENERATOR_SPEC.keys())

    total_slots = sum(len(GENERATOR_SPEC[k]["formats"]) for k in gen_keys)
    per_slot = target // total_slots

    generators = {}
    for key in sorted(gen_keys):
        n_fmts = len(GENERATOR_SPEC[key]["formats"])
        gen_target = per_slot * n_fmts

        entry = {"target": gen_target, "per_format": per_slot}

        if GENERATOR_SPEC[key]["corpus"]:
            items_per_doc = ITEMS_PER_CALL * CHUNKS_PER_DOC
            entry["docs_needed"] = -(-per_slot // items_per_doc)
        else:
            entry["docs_needed"] = None
            entry["api_calls"] = -(-per_slot // ITEMS_PER_CALL)

        generators[key] = entry

    # Absorb rounding remainder into A
    diff = target - sum(g["target"] for g in generators.values())
    if diff and "A" in generators:
        generators["A"]["target"] += diff

    return {"target": target, "generators": generators}


# ---------------------------------------------------------------------------
# OpenAI settings
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2  # seconds, exponential backoff: 2, 4, 8, 16, 32
