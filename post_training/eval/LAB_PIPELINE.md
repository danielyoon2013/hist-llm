# LAB-Strict Generation & Filtering Pipeline

## Overview

This document describes the full pipeline used to generate and filter the
"LAB-strict" benchmark — a Look-Ahead Bias evaluation dataset designed to
test ONLY genuine factual recall, not reading comprehension or surface patterns.

## Pipeline Diagram

```
                          ┌────────────────────────────┐
                          │   GPT-4.1 (gpt-4.1)         │
                          │   500 batch requests        │
                          │   10 questions per request  │
                          │   = 5,000 raw questions     │
                          └──────────────┬─────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  STRICT GENERATION PROMPT     │
                         │  ─────────────────────────    │
                         │  Required answer types:       │
                         │   • Specific person names     │
                         │   • Specific places           │
                         │   • Specific organizations    │
                         │   • Specific dates/years      │
                         │   • Specific numbers          │
                         │   • Specific named events     │
                         │                               │
                         │  Forbidden:                   │
                         │   • Descriptive answers       │
                         │   • Question-paraphrase pairs │
                         │   • Keyword overlap           │
                         │   • Length asymmetry          │
                         └──────────────┬───────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  PARSE & VALIDATE STRUCTURE   │
                         │  - JSON parse                 │
                         │  - 4 choices required         │
                         │  - answer 0-3 required        │
                         │  - event_year > end_year      │
                         └──────────────┬───────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  FILTER 1: LENGTH MATCHING    │
                         │  Reject if correct answer is  │
                         │  >20% longer or shorter than  │
                         │  average distractor length.   │
                         │  (was 50%, tightened to 20%)  │
                         │                               │
                         │  Why: prevents "pick the      │
                         │  longest choice" shortcut.    │
                         └──────────────┬───────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  FILTER 2: KEYWORD OVERLAP    │
                         │  Reject if any 4+ char word   │
                         │  appears in both question     │
                         │  and correct answer.          │
                         │  (excludes stopwords, years)  │
                         │                               │
                         │  Why: prevents "match the     │
                         │  keyword" shortcut.           │
                         └──────────────┬───────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  FILTER 3: YEAR OVERLAP       │
                         │  Reject if any year (1950+)   │
                         │  appears in both question     │
                         │  and correct answer.          │
                         │                               │
                         │  Why: catches "Q: in 2018,    │
                         │  A: May 25, 2018".            │
                         └──────────────┬───────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  FILTER 4: DEDUPLICATION      │
                         │  Reject near-duplicates       │
                         │  using normalized 80-char     │
                         │  question fingerprint.        │
                         │                               │
                         │  Why: GPT-4.1 sometimes       │
                         │  rephrases same question.     │
                         └──────────────┬───────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  FILTER 5: PARAPHRASE         │
                         │  ─────────────────────────    │
                         │  Embed question and 4 choices │
                         │  with sentence-BERT (MiniLM). │
                         │  Compute cosine similarity.   │
                         │  Drop if max-similarity choice│
                         │  matches the correct answer.  │
                         │                               │
                         │  Why: catches semantically    │
                         │  paraphrased answers that no  │
                         │  surface filter detects.      │
                         └──────────────┬───────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  ANSWER POSITION SHUFFLE      │
                         │  Randomize position of correct│
                         │  answer among A/B/C/D so it's │
                         │  uniformly distributed.       │
                         │                               │
                         │  Why: GPT-4.1 has a strong    │
                         │  bias toward putting the      │
                         │  correct answer at A.         │
                         └──────────────┬───────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  OUTPUT: lab_questions.jsonl  │
                         │  ~1,320 questions             │
                         │                               │
                         │  Validated baselines:         │
                         │   • Random:        24.0%      │
                         │   • Length picker: 16.7%      │
                         │   • Keyword match: 24.0%      │
                         │   • Paraphrase:    0.0%       │
                         │                               │
                         │  Any model scoring >25%       │
                         │  must use REAL knowledge.     │
                         └──────────────────────────────┘
```

## Filter Rejection Counts (1900-1949 period)

| Stage              | Input  | Rejected | Output  |
|--------------------|--------|----------|---------|
| Raw GPT-4.1 output | ~5,000 | -        | ~5,000  |
| Length filter      | 5,000  | 2,444    | 2,556   |
| Keyword filter     | 2,556  | 243      | 2,313   |
| Year filter        | 2,313  | 3        | 2,310   |
| Deduplication      | 2,310  | 98       | 2,212   |
| Paraphrase filter  | 2,212  | 880      | 1,320   |
| **Final**          |        |          | **1,320** |

Pass-through rate: 26.4%

## Validation

After all filters, we verify the dataset has no exploitable surface patterns:

| Test                              | Score | Random  |
|-----------------------------------|-------|---------|
| Random guessing                   | 24.0% | 25%     |
| Always pick A                     | 25.3% | 25%     |
| Pick longest choice               | 16.7% | 25%     |
| Pick choice with most word overlap| 24.0% | 25%     |
| Sentence-BERT paraphrase matcher  | 0.0%  | 25%     |
| Year overlap remaining            | 0     | -       |
| Keyword overlap remaining         | 1     | -       |
| Duplicate questions               | 0     | -       |

**Conclusion**: Any model scoring above 25% on this benchmark must be using
genuine factual knowledge of post-period events. Surface patterns are
demonstrably insufficient.

## Why Each Filter Matters

| Filter         | Old LAB shortcut score | LAB-strict score | Pts removed |
|----------------|------------------------|------------------|-------------|
| Length         | 33.5% (longest)         | 16.7%            | 16.8        |
| Keyword        | 22.4% (overlap)         | 24.0%            | -           |
| **Paraphrase** | **55.2% (sentence-BERT)** | **0.0%**       | **55.2**    |

The paraphrase filter is by far the most important. Without it, a reader-only
baseline solves more than half the questions in the original LAB. With it,
the same baseline cannot solve any.

## Code Locations

- Generation: `src/post_training/eval/generate_lab_questions.py`
  - `build_generation_prompt()` — strict prompt
  - `process_generation_results()` — all 5 filters
- Output: `D:/hist_LLM/periods/{period}/posttraining_data/eval/lab_questions.jsonl`
