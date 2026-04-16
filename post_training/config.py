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
DEFAULT_SFT_SIZE = 9_500        # ~1% of train (95% of 1M target)
DEFAULT_TEST_RATIO = 0.05       # 5% holdout for training-loss monitoring

# ---------------------------------------------------------------------------
# Generator specification — single source of truth
# ---------------------------------------------------------------------------
# Allocation is purely determined by format count: each format slot gets
# an equal share of the target. No manual percentages to maintain.
#
# At 1M target with 18 total format slots:
#   per_slot = ~55,556  →  A(3)=166,667  B(3)=166,667  ...  F(3)=166,667
#
# All generators are corpus-based (need document text).
# MC ~33%, non-MC ~67% — reduces position bias from over-training on letter-picking.

ITEMS_PER_CALL = 2      # items requested per API call (all generators)
CHUNKS_PER_DOC = 2      # average chunks per document (6000 chars, 300 overlap)

GENERATOR_SPEC = {
    # collections=None → all collections; list → restrict to those collections only.
    # pass_rate = empirically measured fraction of items that survive format_conversation()
    #             validation; docs_needed and the call cap are inflated by 1/pass_rate so
    #             that the per-slot target is met AFTER filtering.
    # weight    = relative allocation per format slot (default 1.0). Lower weight = fewer
    #             conversations from that generator. Used to dial down generators with
    #             expensive pass rates or marginal benchmark contribution.
    # Gen A targets ARC-style grade-school science.
    # Open-Science-Pile = biology/geology/dredging research papers.
    # USPTO = patents covering circuits, materials, chemistry, mechanical principles
    # (broader topic coverage closer to ARC's physical/life/earth science mix).
    # Gen F at weight=0.6: Winogrande pass rate is 62%, so each Gen F conversation
    # costs ~1.6x the API of other generators. Reducing weight redistributes budget
    # to higher-yield generators (A/B/C/D/E).
    "A": {"formats": ("mc4", "open", "cot"),         "corpus": True, "collections": ["Open-Science-Pile", "USPTO"], "pass_rate": 0.95, "weight": 1.0},
    "B": {"formats": ("mc2", "cot"),                 "corpus": True, "collections": None, "pass_rate": 0.95, "weight": 1.0},
    "C": {"formats": ("mc4_passage", "open", "cot"), "corpus": True, "collections": None, "pass_rate": 0.90, "weight": 1.0},
    "D": {"formats": ("mc4", "open", "cot"),         "corpus": True, "collections": None, "pass_rate": 0.95, "weight": 1.0},
    "E": {"formats": ("mc4", "open", "cot"),         "corpus": True, "collections": None, "pass_rate": 0.85, "weight": 1.0},
    "F": {"formats": ("mc2", "cot"),                 "corpus": True, "collections": None, "pass_rate": 0.62, "weight": 0.6},
}


def compute_plan(target=DEFAULT_TARGET, gen_keys=None):
    """Derive per-generator targets and doc counts from format counts.

    Each generator's docs_needed and effective_target are inflated by 1/pass_rate
    so the post-filter yield meets the per-slot target. `target` in the returned
    entry is the NOMINAL goal (post-filter conversations); `effective_target` is
    what base.py should use for call-cap sizing.

    Returns:
        {
            "target": 1_000_000,
            "generators": {
                "A": {"target": 187500, "per_format": 62500, "pass_rate": 0.95,
                      "effective_target": 197368, "docs_needed": 16447},
                ...
            }
        }
    """
    if gen_keys is None:
        gen_keys = list(GENERATOR_SPEC.keys())

    # Weighted slot allocation: each gen's share = weight × n_formats / total_weighted_slots.
    # Default weight = 1.0; lower weight reduces a gen's allocation.
    total_weighted_slots = sum(
        GENERATOR_SPEC[k].get("weight", 1.0) * len(GENERATOR_SPEC[k]["formats"])
        for k in gen_keys
    )

    generators = {}
    for key in sorted(gen_keys):
        spec = GENERATOR_SPEC[key]
        n_fmts = len(spec["formats"])
        pass_rate = spec.get("pass_rate", 1.0)
        weight = spec.get("weight", 1.0)
        gen_target = int(target * (weight * n_fmts) / total_weighted_slots)
        per_format = gen_target // n_fmts
        effective_target = int(gen_target / pass_rate) if pass_rate < 1.0 else gen_target

        entry = {
            "target": gen_target,
            "per_format": per_format,
            "pass_rate": pass_rate,
            "weight": weight,
            "effective_target": effective_target,
        }

        if spec["corpus"]:
            items_per_doc = ITEMS_PER_CALL * CHUNKS_PER_DOC
            effective_per_format = int(per_format / pass_rate) if pass_rate < 1.0 else per_format
            entry["docs_needed"] = -(-effective_per_format // items_per_doc)
        else:
            entry["docs_needed"] = None
            effective_per_format = int(per_format / pass_rate) if pass_rate < 1.0 else per_format
            entry["api_calls"] = -(-effective_per_format // ITEMS_PER_CALL)

        generators[key] = entry

    # Absorb rounding remainder into A
    diff = target - sum(g["target"] for g in generators.values())
    if diff and "A" in generators:
        generators["A"]["target"] += diff
        generators["A"]["effective_target"] += diff

    return {"target": target, "generators": generators}


# ---------------------------------------------------------------------------
# OpenAI settings
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
# Per-generator overrides. Used for tasks where mini is too weak (e.g. Gen F
# Winogrande-style pronoun resolution, which requires constructing genuine
# ambiguity that mini struggles with).
GENERATOR_MODEL_OVERRIDES = {
    "F": "gpt-4o",
}
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2  # seconds, exponential backoff: 2, 4, 8, 16, 32
