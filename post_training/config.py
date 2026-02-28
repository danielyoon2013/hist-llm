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
METADATA_SHARE = 0.025          # 2.5% each for D (temporal) and H (hist facts)

# Corpus generators' output per chunk (items_per_chunk × num_formats):
#   A=6, B=6, C=6, E=4, F=6, G=2 → 30 examples per chunk, 60 per doc (2 chunks)
EXAMPLES_PER_DOC = 60

def compute_allocation(target=DEFAULT_TARGET):
    """Compute per-generator example targets from a total target.

    Returns dict: {gen_letter: target_examples}

    Corpus generators (A,B,C,E,F,G) share 95% proportionally based on
    their items_per_chunk × num_formats. Metadata generators (D,H) each
    get 2.5%.
    """
    meta_each = int(target * METADATA_SHARE)  # 25,000
    corpus_total = target - 2 * meta_each     # 950,000

    # Per-chunk output for each corpus generator (items_per_chunk × num_formats)
    corpus_weights = {"A": 6, "B": 6, "C": 6, "E": 4, "F": 6, "G": 2}
    weight_sum = sum(corpus_weights.values())  # 30

    alloc = {}
    for gen, weight in corpus_weights.items():
        alloc[gen] = int(corpus_total * weight / weight_sum)
    alloc["D"] = meta_each
    alloc["H"] = meta_each

    # Adjust rounding to hit exact target
    diff = target - sum(alloc.values())
    if diff != 0:
        alloc["A"] += diff  # absorb rounding into largest generator

    return alloc


# ---------------------------------------------------------------------------
# OpenAI settings
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2  # seconds, exponential backoff: 2, 4, 8, 16, 32
