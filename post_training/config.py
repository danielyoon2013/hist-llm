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

    # Input data paths (data/periods_data/{period}/)
    period_data = PROJECT_ROOT / "data" / "periods_data" / period

    # Model data paths (checkpoints, tokenizer, evals)
    model_data = period_data / "model_data"

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
        "metadata_index": period_data / "posttraining_data" / "synthetic" / "document_metadata.parquet",

        # Shared instruct datasets (not period-specific)
        "instruct_data_dir": PROJECT_ROOT / "data" / "instruct_data",

        # Model data paths
        "model_data_dir": model_data,
        "tokenizer_dir": model_data / "tokenizer",
        "base_checkpoints": model_data / "base_checkpoints",
        "mid_checkpoints": model_data / "mid_checkpoints",
        "chatsft_checkpoints": model_data / "chatsft_checkpoints",
        "eval_dir": model_data / "base_eval",
        "report_dir": model_data / "report",

        # External data sources (D: drive)
        "raw_data_root": Path("D:/English"),
        "classified_root": Path("D:/English_Classified"),

        # Additional data sources
        "additional_data_dir": PROJECT_ROOT / "Data" / "additional_data",
        "nyt_filtered_dir": PROJECT_ROOT / "Data" / "additional_data" / "news_archives" / "NYT_filtered_500char",
        "economist_dir": PROJECT_ROOT / "Data" / "additional_data" / "news_archives" / "Economist",
        "ft_dir": PROJECT_ROOT / "Data" / "additional_data" / "news_archives" / "FT",
        "newswire_dir": PROJECT_ROOT / "Data" / "additional_data" / "newswire",
    }

# ---------------------------------------------------------------------------
# OpenAI settings
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2  # seconds, exponential backoff: 2, 4, 8, 16, 32
