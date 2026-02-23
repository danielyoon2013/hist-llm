# Quality Control Pipeline

> **Paper section:** 3.3 (Quality Control)
> **Source code:** `src/post_training/instruct/filter.py`, `split.py`, `synth_config.yaml`
> **Cross-references:** `03_SYNTHETIC_DATA_GENERATORS.md`, `../context_free_qa_writeup.md`, `../instruct_dataset_summary_v2.csv`

---

## 1. Pipeline Overview

Every synthetic training example passes through a multi-stage quality pipeline before entering the final training set.

```
Generated Data (from Generators A-H)
      │
      ▼
[Stage 1] Format Validation          — Valid JSON? Correct schema?
      │
      ▼
[Stage 2] Deduplication              — Exact + near-duplicate removal
      │
      ▼
[Stage 3] LAB Temporal Filtering     — Post-period knowledge detection
      │
      ▼
[Stage 4] Factual Consistency Check  — Answer matches source text?
      │
      ▼
[Stage 5] Difficulty Calibration     — Score and stratify by difficulty
      │
      ▼
[Stage 6] N-gram Decontamination     — Remove eval benchmark overlaps
      │
      ▼
[Stage 7] Train/Test Split           — 95/5 deterministic split
      │
      ▼
Final Training Data
```

**Current implementation status:**
- Stages 1, 3, 7: Fully implemented
- Stage 2: Partially implemented (threshold defined in `synth_config.yaml`)
- Stages 4, 5, 6: Proposed (design specified below)

### Why Quality Control on Our Own Synthetic Data?

Even when generating from our own corpus with GPT-4o-mini, quality filtering is essential. The academic evidence is consistent:

| Paper | Filter Rate | Impact |
|-------|-------------|--------|
| **AlpaGasus** (arXiv:2307.08701) | 83% of Alpaca 52K filtered via ChatGPT scoring | Outperforms full Alpaca; 5.7x faster training |
| **Superfiltering** (arXiv:2402.00530) | 85-95% filtered via IFD scoring | Matches full-data performance at 5-15% of data |
| **Source2Synth** (arXiv:2409.08239) | 13-73% filtered via answerability check | +8-10 accuracy points over unfiltered |
| **Cosmopedia** (HuggingFace, 2024) | 23% topic clusters + MinHash dedup + n-gram decontam | Clean 25B tokens from 30M docs |
| **Self-Instruct** (arXiv:2212.10560) | ROUGE-L dedup + heuristic filters | ~40% filtered from generation pool |
| **Alpaca** (Stanford, **no filter**) | 0% filtered | Community found major quality issues (AlpacaDataCleaned) |
| **LIMA** (arXiv:2305.11206) | Intensive manual curation of 1K examples | Outperforms RLHF-trained models on more data |

**Key pattern:** Aggressive quality filtering (keeping 5-17%) consistently outperforms or matches training on the full unfiltered set, while reducing compute cost 5-7x.

### Two Pipelines: External vs. Corpus-Generated

Quality control serves **different purposes** depending on data source:

| | External Datasets (GSM8K, MATH) | Our Synthetic Data (Generators A-H) |
|---|---|---|
| **Primary risk** | Temporal contamination (post-period knowledge) | Parametric injection (GPT-4o-mini's knowledge leaking beyond source text) |
| **Key filter** | LAB temporal filter (`filter.py`) | Grounding verification (answer entailed by source passage) |
| **Other risks** | Anachronistic content | Malformed JSON, near-duplicates, trivial questions, self-referential responses |
| **Implementation** | Already implemented (Batch API) | Stages 1-2 implemented; 4-6 proposed |

For our synthetic data, we do NOT need LAB filtering (the data is generated from period-specific corpus). Instead, we need **grounding verification**: checking that answers are supported by the source passage and not injected from GPT-4o-mini's parametric knowledge (Sarkar & Vafa, SSRN:4754678).

---

## 2. Stage 1: Format Validation

### What Is Checked

| Check | Rejection Criteria |
|-------|-------------------|
| JSON parsing | Fails to parse as valid JSON |
| Message structure | Missing `role` or `content` fields |
| Conversation format | Not an array of message objects |
| Role alternation | Messages don't alternate user/assistant |
| Non-empty content | Empty string in any message content field |
| Minimum length | Assistant response < 20 characters |

### Implementation

Format validation is embedded in the generation pipeline itself (`run_direct.py`, line 281):
```python
if not item.get("question") or not item.get("answer"):
    continue
```

Conversion to nanochat format (`convert.py`) applies additional structural validation.

### Extension for New Generators

Each new generator (C-H) should include generator-specific schema validation:
- Generator D (Temporal): Verify `level` and `subtype` fields present
- Generator E (Math): Verify `source_numbers` field is non-empty
- Generator H (Anti-Halluc): Verify `event_year` > period end year

---

## 3. Stage 2: Deduplication

### Current Configuration (from `synth_config.yaml`)

```yaml
generation:
  enable_deduplication: true
  similarity_threshold: 0.8
```

### Three Levels of Deduplication

**Level 1 — Exact Match (hash-based):**
- SHA-256 hash of the user message content
- O(1) lookup, catches exact duplicates from overlapping chunks

**Level 2 — Near-Duplicate (similarity-based):**
- ROUGE-L similarity < 0.7 between user messages (following Self-Instruct threshold)
- Applied within each generator type
- More expensive (O(n^2) pairwise) — use MinHash/LSH for large datasets

**Level 3 — Cross-Generator (proposed):**
- The same corpus chunk may generate overlapping questions across generators A, B, C
- Check: if a document chunk produced a QA pair AND a reading comprehension pair with >0.6 similarity, keep only the higher-quality one (by difficulty score)
- This prevents the training set from having near-identical questions in different formats

### Expected Dedup Rates

Based on existing data: overlapping chunks (300 char overlap on 6,000 char chunks = 5% overlap) produce ~2-3% exact duplicates. Near-duplicate rate depends on question diversity.

---

## 4. Stage 3: LAB Temporal Filtering (Implemented)

This is the core quality gate for temporal isolation. It is fully implemented and has been run on the 1950-1999 period.

### Mechanism

For each conversation, GPT-4o-mini classifies whether it requires knowledge that did not exist before the period's end year.

### Classification Prompt (from `filter.py`, lines 81-91)

```
Does the following conversation require knowledge that did NOT exist before
the year {end_year}?

Consider: historical events, technology, people who became famous, scientific
discoveries, cultural works, organizations, and terminology that emerged
after {end_year}.

Conversation:
---
{text}
---

Respond with JSON: {"keep": true} if the conversation only uses knowledge
available before {end_year}, or {"keep": false} if it requires post-{end_year}
knowledge.
```

### Text Extraction

To minimize cost, only the first 2 turns are extracted, truncated to 500 chars per message:
```python
def extract_text_for_classification(messages):
    parts = []
    for msg in messages[:4]:  # first 2 turns max
        parts.append(f"[{msg['role']}]: {msg['content'][:500]}")
    return "\n".join(parts)
```

### Batch API Workflow

```
Step 1: --submit    → Build batch requests, submit to OpenAI Batch API
Step 2: --check     → Poll batch status (~24h processing)
Step 3: --process   → Download results, apply keep/remove decisions
```

- Batch chunk size: 50,000 requests per batch (OpenAI limit)
- Model: GPT-4o-mini (cost: ~50% cheaper than real-time API)
- Conservative default: `keep=True` on JSON parse error

### Outputs

| Output | Location |
|--------|----------|
| Filtered data | `{period}/posttraining_data/final/filtered/` |
| Removed items | `{period}/posttraining_data/final/removed/` (for inspection) |
| Classification scores | `{period}/posttraining_data/LAB_scores/` |

### Filtering Results (1950-1999 Period)

From `../instruct_dataset_summary_v2.csv`:

| Dataset | Category | Total | Removed | Kept | Contamination Rate |
|---------|----------|-------|---------|------|-------------------|
| GSM8K | Math | 7,473 | 188 | 7,285 | 2.5% |
| MATH | Math | 7,500 | 23 | 7,477 | 0.3% |
| AIME/AMC | Math | 4,069 | 26 | 4,043 | 0.6% |
| FOLIO | Logic | 1,001 | 284 | 717 | 28.4% |
| ScienceQA | Science | 5,837 | 131 | 5,706 | 2.2% |
| ARC-Easy | Science | 2,251 | 38 | 2,213 | 1.7% |
| ARC-Challenge | Science | 1,119 | 18 | 1,101 | 1.6% |
| SmolTalk | General | 460,341 | 149,970 | 310,371 | **32.6%** |
| MMLU | General | 99,842 | 34,518 | 65,324 | **34.6%** |
| LogiQA | Logic | 7,376 | 3,057 | 4,319 | 41.4% |
| PIQA | Commonsense | 16,113 | 1,672 | 14,441 | 10.4% |
| HotpotQA | Multi-hop | 90,447 | 51,220 | 39,227 | **56.6%** |
| MuSiQue | Multi-hop | 19,938 | 8,259 | 11,679 | 41.4% |
| HumanEval | Code | 164 | 3 | 161 | 1.8% |
| CodeContests | Code | 13,134 | 5,355 | 7,779 | 40.8% |

**Key insight:** Math datasets (GSM8K, MATH) have near-zero contamination because math is timeless. General/multi-hop datasets (SmolTalk, MMLU, HotpotQA) have 30-57% contamination. This validates the decision to move to fully synthetic data from our own corpus.

### Extension for Corpus Q&A

The same pipeline supports corpus-generated QA via the `--corpus` flag:
```bash
python -m src.post_training.instruct.filter --period 1950_1999 --submit --corpus
```

For fully synthetic data, contamination rates should be near-zero (since the source documents are period-specific), but the filter still catches edge cases where GPT-4o-mini hallucinated modern knowledge into generated QA pairs.

---

## 5. Stage 4: Factual Consistency Check (Proposed)

### Purpose

Verify that generated answers are actually supported by the source text, not hallucinated by GPT-4o-mini during generation.

### Approach

For each QA pair, compare the generated answer against the source document chunk:

```
CONSISTENCY_PROMPT = """Given the source text and a question-answer pair generated
from it, determine if the answer is factually consistent with the source.

Source text:
{source_text}

Question: {question}
Answer: {answer}

Is the answer factually supported by the source text?
Respond with JSON: {"consistent": true/false, "reason": "brief explanation"}"""
```

### Implementation Notes

- Run via Batch API (same 3-step workflow as LAB filtering)
- Requires storing source document chunk IDs alongside generated QA pairs
- For Generator A (existing QA): chunk → QA mapping is tracked in `run_direct.py`
- For new generators: extend the output format to include `source_chunk_id`
- Reject pairs where `consistent: false`

### Expected Impact

Based on GPT-4o-mini hallucination rates in structured generation (~5-10%), expect to remove 3-7% of generated pairs. This is a worthwhile quality improvement, especially for factual QA (Generator A) and reading comprehension (Generator C).

---

## 6. Stage 5: Difficulty Calibration (Proposed)

### Purpose

Score each QA pair by difficulty level to ensure the training mix includes easy, medium, and hard examples. Without calibration, GPT-4o-mini tends to generate medium-difficulty questions, under-representing both extremes.

### Rating Prompt (from `synth_config.yaml`, adapted)

```yaml
qa_rating: |
  Rate each question-answer pair on a scale from 1-10, based on:
  - Accuracy (0-3): factual correctness
  - Relevance (0-2): relevance to content
  - Clarity (0-2): clear language
  - Usefulness (0-3): value for model learning
```

### Difficulty Tiers

| Tier | Score Range | Target % of Training Set |
|------|-------------|--------------------------|
| Easy | 1-3 | 20% |
| Medium | 4-7 | 60% |
| Hard | 8-10 | 20% |

### Implementation Notes

- Run quality scoring via Batch API on the filtered dataset
- Use the `curate.threshold` (7.0 from `synth_config.yaml`) as a minimum quality floor
- If the difficulty distribution is skewed, over-sample from under-represented tiers or use Evol-Instruct to increase difficulty of easy examples (see `03_SYNTHETIC_DATA_GENERATORS.md`, Section 7)

---

## 7. Stage 6: N-gram Decontamination (Proposed)

### Purpose

Ensure that training data does not contain exact or near-exact matches with evaluation benchmark questions. This prevents inflated benchmark scores.

### Approach (following Phi-4, arXiv:2412.08905)

1. **13-gram matching** against all evaluation benchmarks (MMLU, ARC, GSM8K test sets, LAB eval questions)
2. **7-gram matching** for shorter questions where 13-gram is too strict
3. Remove any training example that has a 13-gram or 7-gram match with any eval question

### Implementation

```python
def build_ngram_index(eval_questions, n=13):
    """Build a set of all n-grams from evaluation questions."""
    ngrams = set()
    for q in eval_questions:
        tokens = q.lower().split()
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i:i+n]))
    return ngrams

def is_contaminated(text, ngram_index, n=13):
    """Check if text contains any n-gram from the eval set."""
    tokens = text.lower().split()
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i:i+n]) in ngram_index:
            return True
    return False
```

### Eval Sets to Decontaminate Against

- All `nanochat/tasks/*.py` evaluation data
- LAB eval questions (`{period}/posttraining_data/eval/lab_questions.jsonl`)
- Any held-out test split from the train/test split (Stage 7)

---

## 8. Stage 7: Train/Test Split (Implemented)

### Configuration (from `split.py`)

| Parameter | Value |
|-----------|-------|
| Test ratio | 5% (0.05) |
| Random seed | 42 (deterministic) |
| Shuffle | Yes (via `random.Random(seed)`) |

### Workflow

```bash
python -m src.post_training.instruct.split --period 1950_1999
```

Reads from `final/filtered/`, writes to `final/train/` and `final/test/`.

### Output Structure

```
{period}/posttraining_data/final/
├── filtered/          # Input: LAB-filtered files
├── removed/           # Removed items (for inspection)
├── train/             # 95% training splits
│   ├── hist_economist_train.jsonl
│   ├── hist_nyt_filtered_train.jsonl
│   ├── smoltalk_filtered_train.jsonl
│   └── ... (40 files)
└── test/              # 5% test splits
    ├── hist_economist_test.jsonl
    └── ...
```

### Current Volume (1950-1999)

- Total training examples: 838,313
- Total test examples: ~44,000
- Files: 40 per split

---

## 9. Quality Metrics and Reporting

### Metrics to Track per Stage

| Stage | Metric | How to Report |
|-------|--------|---------------|
| Format Validation | Rejection rate by generator type | Table: generator × rejection count |
| Deduplication | Duplicate rate (exact + near) | Percentage removed, by generator |
| LAB Filtering | Contamination rate by source | Table (see Section 4 above) |
| Factual Consistency | Inconsistency rate by generator | Percentage flagged as inconsistent |
| Difficulty Calibration | Distribution histogram | Easy/Medium/Hard percentages |
| N-gram Decontamination | Matches found by benchmark | Count per benchmark |
| Train/Test Split | Final dataset sizes | Total examples, by source |

### Paper Table Template

**Table X: Quality Pipeline Statistics (Period: 1950-1999)**

| Stage | Input | Removed | Output | Rate |
|-------|-------|---------|--------|------|
| Raw generated | — | — | X | — |
| Format validation | X | Y | Z | Y/X% |
| Deduplication | Z | ... | ... | ...% |
| LAB filtering | ... | ... | ... | ...% |
| Consistency check | ... | ... | ... | ...% |
| N-gram decontam | ... | ... | ... | ...% |
| **Final** | — | — | **N** | — |

---

## 10. SFT Alignment Justification

Our synthetic QA pairs are presented to the model without the source document at training time. This is standard practice, well justified by the LIMA hypothesis (Zhou et al., 2023):

> "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users."

Since our QA pairs are generated from the same corpus the model was pre-trained on, no new knowledge is introduced during SFT — only new formats.

See `../context_free_qa_writeup.md` for the full analysis with citations (LIMA, Alpaca, Llama 2, Gekhman et al.).

---

## References

- `src/post_training/instruct/filter.py` — LAB filtering implementation
- `src/post_training/instruct/split.py` — Train/test split
- `src/post_training/corpus/synth_config.yaml` — Generation and curation parameters
- `../instruct_dataset_summary_v2.csv` — Full contamination analysis
- `../context_free_qa_writeup.md` — SFT alignment justification
- Abdin et al. (2024). Phi-4 Technical Report. arXiv:2412.08905 (n-gram decontamination)
- Wang et al. (2022). Self-Instruct. arXiv:2212.10560 (ROUGE-L dedup threshold)
- Zhou et al. (2023). LIMA. NeurIPS 2023 (superficial alignment hypothesis)
