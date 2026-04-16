"""
Answerability classifier for benchmark questions.

Labels each question as one of:
  - "answerable"    — can be answered using only period-consistent knowledge
  - "unanswerable"  — requires post-period knowledge/content (e.g., iPhone, Internet)
  - "ambiguous"     — borderline; LLM was unsure

Strategy: LLM judge only. A single prompt asks gpt-4o-mini whether the
question's content fits the period. Results are cached to disk per-benchmark
to avoid re-calling the API on reruns.
"""

import json
import hashlib
from pathlib import Path
from typing import Iterable

from src.post_training.config import load_api_key

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_LLM_PROMPT = """You are evaluating whether a multiple-choice question can be answered using only knowledge and commonsense that was available during {start_year}-{end_year}.

Question:
{question}

Classify the question as:
- "answerable": The question's content, concepts, and required knowledge were all available in {start_year}-{end_year}. A well-educated person of that era could in principle answer it. Examples: gravity, plant biology, cooking, farming, simple arithmetic, period-appropriate history.
- "unanswerable": The question requires post-{end_year} knowledge, technology, cultural references, or events (e.g., computers, Internet, smartphones, post-WWII politics, genetic engineering, space exploration after 1949).
- "ambiguous": Borderline — the question might be answerable but relies on cultural context that may not exist or may be different.

Focus on whether the CONTENT of the question (the scenario, vocabulary, required knowledge) fits the period, not whether a reader from that era would agree with the "correct" answer.

Return ONLY a JSON object:
{{"label": "answerable" | "unanswerable" | "ambiguous", "reason": "one short sentence"}}"""


def classify_llm(text: str, start_year: int = 1900, end_year: int = 1949,
                 model: str = "gpt-4o-mini") -> dict:
    """
    LLM-based answerability classifier. Returns {label, reason}.

    Requires an OpenAI API call. Caller should cache results.
    """
    from openai import OpenAI
    client = OpenAI(api_key=load_api_key())

    prompt = _LLM_PROMPT.format(start_year=start_year, end_year=end_year, question=text)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0.0,
    )
    body = resp.choices[0].message.content
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {"label": "ambiguous", "reason": "llm_json_parse_failed"}
    label = parsed.get("label", "ambiguous")
    if label not in ("answerable", "unanswerable", "ambiguous"):
        label = "ambiguous"
    return {"label": label, "reason": parsed.get("reason", "")[:200]}


# ---------------------------------------------------------------------------
# Hybrid + batch with caching
# ---------------------------------------------------------------------------

def _hash_question(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def classify_batch(questions: Iterable[dict], cache_path: Path,
                   start_year: int = 1900, end_year: int = 1949,
                   max_workers: int = 32, verbose: bool = True) -> list[dict]:
    """
    Classify a batch of questions using LLM judge only, with on-disk caching.

    Args:
      questions: iterable of dicts with at least {"index": int, "question": str}
      cache_path: JSONL file for caching; rows: {qhash, index, question, label, reason}
      start_year, end_year: period bounds
      max_workers: parallel LLM calls

    Returns:
      list of classification dicts aligned with input order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in open(cache_path, "r", encoding="utf-8"):
            row = json.loads(line)
            cache[row["qhash"]] = row

    questions = list(questions)
    results: list[dict] = [None] * len(questions)
    to_llm: list[tuple[int, str, str]] = []

    # Cache lookup first
    for i, q in enumerate(questions):
        text = q["question"]
        qhash = _hash_question(text)
        if qhash in cache:
            results[i] = {**cache[qhash], "index": q.get("index", i)}
        else:
            to_llm.append((i, qhash, text))

    if verbose:
        n_cached = len(questions) - len(to_llm)
        print(f"  cached: {n_cached} | need LLM: {len(to_llm)}")

    def _judge(args):
        i, qhash, text = args
        res = classify_llm(text, start_year, end_year)
        entry = {
            "qhash": qhash, "index": questions[i].get("index", i), "question": text[:400],
            "label": res["label"], "reason": res["reason"],
        }
        return i, entry

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_judge, args): args for args in to_llm}
        for fut in as_completed(futures):
            i, entry = fut.result()
            results[i] = entry
            cache[entry["qhash"]] = entry
            completed += 1
            if verbose and completed % 100 == 0:
                print(f"    {completed}/{len(to_llm)} done")

    # Write cache
    with open(cache_path, "w", encoding="utf-8") as f:
        for row in cache.values():
            f.write(json.dumps(row) + "\n")

    return results


def load_classifications(cache_path: Path) -> dict[str, dict]:
    """Load an existing classification cache keyed by question hash."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    out = {}
    for line in open(cache_path, "r", encoding="utf-8"):
        row = json.loads(line)
        out[row["qhash"]] = row
    return out
