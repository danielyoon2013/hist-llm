"""
Metrics and display utilities for benchmark error analysis.

Used together with answerability.classify_batch() to produce:
  - stratified accuracy triples
  - confidence breakdowns (right vs wrong on each answerability subset)
  - high-confidence wrong examples for qualitative review
  - confidence histograms
"""

import re
import json
from pathlib import Path
import pandas as pd


def load_details(path: Path) -> pd.DataFrame:
    """Load a *_details.jsonl file into a DataFrame. Empty DataFrame if missing."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    df = pd.DataFrame(rows)
    if "question" in df.columns:
        df["question"] = df["question"].astype(str)
    return df


_MC_CHOICE_RE = re.compile(r"-\s*(.+?)=([A-Z])\s*\n", re.MULTILINE)


def extract_stem_and_choices(question_text: str) -> dict:
    """
    Parse the nanochat MC question format into {stem, choices: {A: ..., B: ..., ...}}.

    Format:
      Multiple Choice question: STEM
      - choice_1=A
      - choice_2=B
      ...
      Respond only with the letter of the correct answer.
    """
    stem = question_text
    m = re.search(r"Multiple Choice question:\s*(.+?)(?:\n-\s|\Z)",
                  question_text, re.DOTALL)
    if m:
        stem = m.group(1).strip()

    choices = {}
    for match in _MC_CHOICE_RE.finditer(question_text + "\n"):
        text, letter = match.group(1).strip(), match.group(2)
        choices[letter] = text

    return {"stem": stem, "choices": choices}


def accuracy_triple(df: pd.DataFrame, classifications: dict[str, dict]) -> dict:
    """
    Compute (overall, answerable, unanswerable, ambiguous) accuracy.

    df: DataFrame from load_details
    classifications: dict keyed by qhash -> {label, reason, ...}
    """
    from src.post_training.analysis.answerability import _hash_question

    if df.empty:
        return {}

    df = df.copy()
    df["qhash"] = df["question"].map(_hash_question)
    df["label"] = df["qhash"].map(lambda h: classifications.get(h, {}).get("label", "ambiguous"))

    def bucket_acc(subset):
        if len(subset) == 0:
            return {"n": 0, "correct": 0, "acc": None}
        return {
            "n": len(subset),
            "correct": int(subset["correct"].sum()),
            "acc": float(subset["correct"].mean()),
        }

    return {
        "overall": bucket_acc(df),
        "answerable": bucket_acc(df[df["label"] == "answerable"]),
        "unanswerable": bucket_acc(df[df["label"] == "unanswerable"]),
        "ambiguous": bucket_acc(df[df["label"] == "ambiguous"]),
    }


def confidence_breakdown(df: pd.DataFrame, classifications: dict[str, dict]) -> pd.DataFrame:
    """
    Return a DataFrame with per-question label and correct/wrong-confidence split.
    Columns: index, qhash, label, correct, confidence, confidence_on_wrong
    """
    from src.post_training.analysis.answerability import _hash_question

    df = df.copy()
    df["qhash"] = df["question"].map(_hash_question)
    df["label"] = df["qhash"].map(lambda h: classifications.get(h, {}).get("label", "ambiguous"))
    return df[["index", "qhash", "label", "correct", "confidence", "predicted", "expected"]]


def high_confidence_wrong(df: pd.DataFrame, classifications: dict[str, dict],
                          label_filter: str = "answerable", n: int = 10,
                          min_conf: float = 0.5) -> pd.DataFrame:
    """
    Top-N highest-confidence wrong answers matching label_filter.
    """
    from src.post_training.analysis.answerability import _hash_question

    df = df.copy()
    df["qhash"] = df["question"].map(_hash_question)
    df["label"] = df["qhash"].map(lambda h: classifications.get(h, {}).get("label", "ambiguous"))
    df["reason"] = df["qhash"].map(lambda h: classifications.get(h, {}).get("reason", ""))

    wrong = df[(df["correct"] == False) & (df["confidence"] >= min_conf)]
    if label_filter:
        wrong = wrong[wrong["label"] == label_filter]
    return wrong.sort_values("confidence", ascending=False).head(n)


def plot_confidence_histogram(df: pd.DataFrame, classifications: dict[str, dict],
                              title: str = "Confidence distribution",
                              ax=None):
    """Plot confidence distribution split by (correct × label)."""
    import matplotlib.pyplot as plt
    from src.post_training.analysis.answerability import _hash_question

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    df = df.copy()
    df["qhash"] = df["question"].map(_hash_question)
    df["label"] = df["qhash"].map(lambda h: classifications.get(h, {}).get("label", "ambiguous"))

    colors = {
        ("answerable", True):   "#2ecc71",
        ("answerable", False):  "#e74c3c",
        ("unanswerable", True): "#95a5a6",
        ("unanswerable", False): "#34495e",
        ("ambiguous", True):    "#f1c40f",
        ("ambiguous", False):   "#e67e22",
    }
    for (label, correct), sub in df.groupby(["label", "correct"]):
        if len(sub) == 0:
            continue
        ax.hist(sub["confidence"], bins=40, range=(0, 1), alpha=0.55,
                label=f"{label} {'right' if correct else 'wrong'} (n={len(sub)})",
                color=colors.get((label, correct), "#bdc3c7"))
    ax.set_xlabel("Model confidence (max softmax probability)")
    ax.set_ylabel("Question count")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    return ax
