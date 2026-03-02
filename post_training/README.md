# Post-Training Pipeline

Synthetic data generation and assembly for historical LLM post-training (6 periods, 1678-2023).

## Three-Step Pipeline

```bash
# Step 1: Prepare source documents (one-time per period)
python -m src.post_training.prepare --period 1900_1949

# Step 2: Generate synthetic data
python -m src.post_training.generate submit --period 1900_1949     # batch API (50% off)
python -m src.post_training.generate check  --period 1900_1949     # poll status
python -m src.post_training.generate process --period 1900_1949    # download + write

# Step 3: Assemble into train/test splits
python -m src.post_training.assemble --period 1900_1949
```

For testing, use sync mode with a small target:
```bash
python -m src.post_training.generate --period 1900_1949 --target 120 --sync
```

---

## How Allocation Works

One parameter — `--target` (default: 1,000,000) — drives the entire pipeline.

### The Rule: Equal Weight Per Format Slot

Each generator produces 1+ format variants. Every format slot gets an equal share of the target. A generator's allocation is simply `per_slot x number_of_formats`.

```
GENERATOR_SPEC in config.py:

Gen   Name              Formats                       Slots
───   ────              ───────                       ─────
A     Factual QA        mc4, open                     2
B     Chain-of-Thought  mc4, cot                      2
C     Comprehension     mc4_passage                   1
D     Quantitative      open, cot                     2
E     Completion        mc4                           1
F     Instruct          mc4_passage                   1
                                              Total:  9 slots
```

At `--target 1,000,000`:

```
per_slot = 1,000,000 / 9 = 111,111

A: 111,111 x 2 = 222,222
B: 111,111 x 2 = 222,222
C: 111,111 x 1 = 111,111
D: 111,111 x 2 = 222,222
E: 111,111 x 1 = 111,111
F: 111,111 x 1 = 111,111
                ─────────
         Total: 999,999  (+1 absorbed into A = 1,000,000)
```

### From Target to Docs Needed

Each document is chunked (~2 chunks per doc at 6000 chars). Each chunk produces 2 items per API call (`ITEMS_PER_CALL = 2`). So:

```
items_per_doc = ITEMS_PER_CALL x CHUNKS_PER_DOC = 2 x 2 = 4

docs_needed (per generator) = ceil(per_slot / items_per_doc)
                             = ceil(111,111 / 4) = 27,778
```

Each generator independently samples its own docs from `synthetic/input/`. All of this is computed by `compute_plan()` — you only specify `--target`.

### Per-Period Output Targets

| Output | Size | Description |
|--------|-----:|-------------|
| Mid-train | 1,000,000 | All synthetic examples |
| SFT | 10,000 | 1% proportional subsample |
| Test | ~50,000 | 5% holdout for training-loss monitoring |

---

## The 6 Generators (9 Format Slots)

All generators are corpus-based — they read documents from `synthetic/input/` and produce training conversations grounded in historical text.

| ID | Name | Formats | Benchmark Alignment | Items/Call |
|----|------|---------|---------------------|-----------|
| A | Factual QA | `mc4`, `open` | MMLU, ARC | 2 |
| B | Chain-of-Thought | `mc4`, `cot` | ARC, GSM8K | 2 |
| C | Reading Comprehension | `mc4_passage` | RACE | 2 |
| D | Quantitative | `open`, `cot` | GSM8K | 2 |
| E | Historical Completion | `mc4` | HellaSwag | 2 |
| F | Instruction Following | `mc4_passage` | RACE | 2 |

### Format Key

| Format | Description | Eval Benchmarks |
|--------|-------------|-----------------|
| `mc4` | 4-choice MC | MMLU, ARC, HellaSwag |
| `mc4_passage` | Passage + 4-choice MC (RACE-style) | RACE |
| `open` | Open-ended Q&A | GSM8K (generative) |
| `cot` | `<think>` reasoning + answer | GSM8K (with reasoning) |

### Multi-Format Rendering

Each generator makes **one API call** per chunk that returns all fields needed for every format. The prompt requests the correct answer, distractors, and reasoning steps together. The same raw response is rendered into multiple formats locally — zero extra API cost.

**Self-contained questions:** Prompts for non-passage generators (A, B, D, E) instruct GPT to produce questions answerable without the source text. Passage-based generators (C, F) include the source passage in the conversation.

**MC answer positioning:** Per-format cyclic counters place the correct answer at A, B, C, D in rotation — guaranteeing ~25% uniform distribution.

---

## Step 1: Prepare Source Documents

`prepare.py` exports documents from two sources into uniform parquets at `synthetic/input/{collection}.parquet` (columns: `doc_name`, `text`).

### Main Corpus (~26 collections)
- Joins quality scores (`corpus/classified/classified_{year}.parquet`) with collection labels from raw parquets
- Applies quality percentile filter (default: top 50%)
- Caps at 10K docs per collection

### Additional Data (4 news archives)
- NYT, Economist, FT, Newswire loaded from `additional_data/raw/`
- Filters by minimum text length (200 chars)
- Joins pre-calculated quality scores from `additional_data/classified/{collection}/classified_{year}.parquet`
- Same quality percentile filter and 10K cap

Quality scores for additional data are generated by `src/base_training/quality/classify_additional.py`, which applies the same corpus-trained Ridge models via BGE embeddings.

```bash
python -m src.post_training.prepare --period 1900_1949
python -m src.post_training.prepare --period 1900_1949 --source corpus       # corpus only
python -m src.post_training.prepare --period 1900_1949 --source additional   # news only
python -m src.post_training.prepare --period 1900_1949 --quality-percentile 75  # top 25%
```

### Step 2: Generate Synthetic Data

```bash
# Batch API (production) — 50% cost savings, ~24h turnaround
python -m src.post_training.generate submit  --period 1900_1949
python -m src.post_training.generate check   --period 1900_1949
python -m src.post_training.generate process --period 1900_1949

# Sync mode (testing) — instant results
python -m src.post_training.generate --period 1900_1949 --target 120 --sync

# Specific generators only
python -m src.post_training.generate submit --period 1900_1949 --generators A B D
```

**Output:** `synthetic/by_generator/gen_{name}_{fmt}.jsonl`

Each line is a nanochat CustomJSON conversation:
```jsonl
[{"role":"user","content":"What caused...?"},{"role":"assistant","content":"The main cause..."}]
```

### Step 3: Assemble

Merges all generator outputs into three files:

```bash
python -m src.post_training.assemble --period 1900_1949 --source raw
python -m src.post_training.assemble --period 1900_1949 --dry-run  # preview only
```

**Output:**
- `final/train/hist_synthetic_midtrain.jsonl` — all examples (1M)
- `final/train/hist_synthetic_sft.jsonl` — 10K proportional subsample
- `final/test/hist_synthetic_test.jsonl` — 5% holdout

---

## API Usage and Cost

Generation uses `gpt-4o-mini` via the OpenAI Batch API (50% discount).

At 1M target with 9 format slots and ~27,778 docs per generator:
- ~6 generators x ~27,778 docs x ~2 chunks = ~333,336 API calls
- At ~$0.00075/call (Batch API average) = **~$250/period**
- x 6 periods = **~$1,500 total**

For testing, use `--target 120 --sync` (< $0.01).

---

## 6 Periods

| Period | Era | Years |
|--------|-----|-------|
| `1678_1849` | Early Modern + Industrial Revolution | 1678-1849 |
| `1850_1899` | Victorian Era | 1850-1899 |
| `1900_1949` | World Wars Era | 1900-1949 |
| `1950_1999` | Cold War + Digital Age | 1950-1999 |
| `2000_2009` | Early Internet | 2000-2009 |
| `2010_2023` | Social Media Era | 2010-2023 |

---

## File Reference

```
post_training/
├── config.py                          # Central config: GENERATOR_SPEC, compute_plan(), paths
├── prepare.py                         # Step 1: Export corpus + additional → uniform parquets
├── generate.py                        # Step 2: Submit/check/process batch generation (+ --sync)
├── assemble.py                        # Step 3: Merge generators into train/test splits
├── utils.py                           # Shared: OpenAI client, JSONL I/O, Batch API helpers
│
├── generators/                        # Synthetic data generators (A-F)
│   ├── __init__.py                    # Registry: {"A": GenAFactual, ..., "F": GenFInstruct}
│   ├── base.py                        # BaseGenerator: run(), batch submit/process, render_mc
│   ├── prompts.py                     # 6 prompt templates (with distractor requests)
│   ├── gen_a_factual.py               # A: Factual QA (mc4, open)
│   ├── gen_b_cot.py                   # B: Chain-of-Thought (mc4, cot)
│   ├── gen_c_comprehension.py         # C: Reading Comprehension (mc4_passage)
│   ├── gen_d_quantitative.py          # D: Quantitative / Math (open, cot)
│   ├── gen_e_completion.py            # E: Sentence Completion (mc4)
│   └── gen_f_instruct.py             # F: Instruction Following (mc4_passage)
│
├── quality/                           # Quality pipeline (deferred for v1)
│   ├── validate.py                    # Format + content validation
│   ├── dedup.py                       # 3-level dedup (exact, MinHash, cross-gen)
│   └── pipeline.py                    # Orchestrator: validate -> dedup
│
├── corpus/                            # Legacy scripts (deprecated, kept for reference)
│   ├── build_index.py                 # Deprecated → use prepare.py
│   ├── export.py                      # Deprecated → use prepare.py
│   └── export_additional.py           # Deprecated → use prepare.py
│
└── eval/                              # Evaluation
    ├── generate_lab_questions.py       # Generate LAB eval MC questions
    └── shuffle_lab_answers.py          # Randomize answer positions
```

---

## Data Directory Structure

```
D:\hist_LLM\periods\{period}\posttraining_data\
│
├── synthetic/
│   ├── input/                         # Source corpus (Step 1: prepare.py)
│   │   ├── {collection}.parquet       # Both corpus and additional data as parquets
│   │   │                              #   (columns: doc_name, text; 10K cap each)
│   │   └── {collection}/              # Legacy: individual txt files (still supported)
│   │       └── doc_00000.txt
│   │
│   ├── by_generator/                  # Per-generator raw output (Step 2: generate.py)
│   │   ├── gen_a_factual_mc4.jsonl
│   │   ├── gen_a_factual_open.jsonl
│   │   ├── gen_b_cot_mc4.jsonl
│   │   ├── gen_b_cot_cot.jsonl
│   │   ├── gen_c_comprehension_mc4_passage.jsonl
│   │   ├── gen_d_quantitative_open.jsonl
│   │   ├── gen_d_quantitative_cot.jsonl
│   │   ├── gen_e_completion_mc4.jsonl
│   │   └── gen_f_instruct_mc4_passage.jsonl
│   │
│   └── batch_temp/                    # Batch API temporary files
│       ├── gen_*_requests.jsonl
│       ├── gen_*_manifest.jsonl
│       ├── gen_*_batch_id.txt
│       └── gen_*_results.jsonl
│
├── final/
│   ├── train/                         # Final train splits (Step 3: assemble.py)
│   │   ├── hist_synthetic_midtrain.jsonl  # All synthetic (1M target)
│   │   └── hist_synthetic_sft.jsonl       # 1% proportional subsample (10K)
│   └── test/
│       └── hist_synthetic_test.jsonl      # 5% holdout
│
└── quality/                           # Quality pipeline outputs (deferred for v1)
    ├── validated/
    └── deduped/
```

---

## nanochat Integration

All output files are in nanochat's **CustomJSON** format:

```jsonl
[{"role":"user","content":"What caused...?"},{"role":"assistant","content":"The main cause..."}]
[{"role":"user","content":"Multiple Choice question: ...\n- Option A=A\n- Option B=B\n..."},{"role":"assistant","content":"B"}]
[{"role":"user","content":"Explain..."},{"role":"assistant","content":"<think>\nStep 1...\n</think>\nThe answer is..."}]
```

---

## Evaluation and Train/Test Strategy

Our evaluation uses a 3-tier framework, with a train/test split designed to serve both training-loss monitoring and generator-level ablation.

### Train/Test Split

`assemble.py` splits all synthetic output into three files:

| Output | Size | Purpose |
|--------|-----:|---------|
| `hist_synthetic_midtrain.jsonl` | ~1,000,000 | Mid-training: all synthetic examples |
| `hist_synthetic_sft.jsonl` | 10,000 | SFT: 1% proportional subsample across generators |
| `hist_synthetic_test.jsonl` | ~50,000 | 5% holdout for training-loss monitoring |

The 5% test holdout is stratified by generator — each generator contributes 5% of its output to the test set. This means the test set reflects the same format/content distribution as the training data, enabling per-generator loss tracking.

**What the test set is for:** Monitoring training loss (is the model learning from each generator's data?). It is NOT used for capability evaluation — that's what the external benchmarks below are for.

**What the test set is NOT for:** Measuring whether the model "knows history" or "can reason." The test set is synthetic data in the same distribution as training; low loss just means the model fits our synthetic format, not that it's actually capable.

### 3-Tier Evaluation Framework

#### Tier 1: Core Capabilities (does the model work as a general LLM?)

Standard benchmarks already implemented in nanochat. These measure fundamental LLM abilities and are used for cross-period comparison since they are temporally neutral.

| Benchmark | Format | Size | What It Measures | Generator Alignment |
|-----------|--------|-----:|------------------|---------------------|
| MMLU | MC-4 | ~16K | Breadth of knowledge (57 subjects) | A (Factual) |
| ARC-Challenge | MC-4 | 1,119 | Scientific reasoning | A, B (Reasoning) |
| GSM8K | Open-ended | 7,473 | Multi-step math reasoning | D (Quantitative) + external GSM8K |
| HellaSwag | MC-4 | ~40K | Commonsense / language modeling | E (Completion) |

**MMLU and ARC caveat:** These contain post-period knowledge. We LAB-filter them per period, but the surviving subset size varies (e.g., ~6K for 1678-1849 vs. ~15K for 2010-2023). Cross-period accuracy comparisons on different-sized subsets are not apples-to-apples. Report as supplementary only.

**GSM8K and HellaSwag** are temporally neutral (math doesn't change; commonsense doesn't change) — safe for direct cross-period comparison.

#### Tier 2: Breadth Capabilities (does domain specialization hurt general ability?)

Held-out benchmarks — NOT included in training data. Any score above random indicates transfer, not memorization.

| Benchmark | Format | What It Measures | Generator Alignment |
|-----------|--------|------------------|---------------------|
| BoolQ | MC-2 + Passage | Boolean reasoning, reading comprehension | A, C |
| PIQA | MC-2 | Physical intuition, common sense | E (Completion) |
| WinoGrande | MC-2 | Coreference resolution | E (Completion) |
| RACE | MC-4 + Passage | Long-passage comprehension | B, C, F |
| SpellingBee | Generative | Lexical knowledge | — |
| Dyck Language | Generative | Symbolic reasoning (bracket matching) | — |

#### Tier 3: Diagnostic — Temporal Isolation (the core thesis)

The most important tier. A model that scores well on Tiers 1-2 but fails Tier 3 has lookahead bias, defeating the project's purpose.

| Metric | Format | Size | What It Measures | Target |
|--------|--------|-----:|------------------|--------|
| LAB Eval | MC-4 | 5,000/period | Post-period knowledge (should be at chance) | ~25% accuracy |
| LAP Score | Scalar | — | `(LAB_accuracy - 0.25) / 0.75` | < 0.05 |

**LAB Eval:** 5,000 MC questions per period about events that occurred AFTER the period's end year. Generated via GPT-4.1 across 10 domains (politics, technology, science, culture, sports, economics, medicine, space, environment, social movements). A perfectly isolated model scores 25% (random chance on 4-choice MC).

**LAP Score interpretation:**

| LAP | Meaning |
|-----|---------|
| 0.00 | Perfect isolation — random chance on future questions |
| 0.00-0.05 | Minimal leakage — acceptable |
| 0.05-0.15 | Moderate leakage — investigate source |
| 0.15-0.30 | Significant leakage — temporal isolation compromised |
| > 0.30 | Severe leakage — substantial future knowledge |

### Generator-to-Evaluation Alignment

Every generator targets specific benchmarks. This enables clean ablation studies: remove a generator, measure which benchmarks degrade.

```
                 Tier 1 (Core)         Tier 2 (Breadth)      Tier 3
                 MMLU ARC GSM8K Hella  BoolQ PIQA Wino RACE  LAB
Gen A (Factual)  **   **               *
Gen B (CoT)           *                               **
Gen C (Compreh)                        *               **
Gen D (Quantit)            **
Gen E (Complet)                  **          *    *
Gen F (Instruct)                                      .
External: GSM8K             **
External: MATH              *

** = primary   * = secondary   . = indirect
```

### When to Evaluate

| Training Stage | What to Evaluate | Purpose |
|----------------|------------------|---------|
| After base training | Tier 1 + LAB Eval | Baseline + temporal isolation check |
| After mid-training | Tier 1 + Tier 2 + LAB Eval | Did mid-training help or hurt? |
| After SFT | All 3 tiers | Full evaluation of final model |

### Results Template

| Period | MMLU | ARC-C | GSM8K | HellaSwag | BoolQ | PIQA | WinoGr | RACE | LAB | LAP |
|--------|------|-------|-------|-----------|-------|------|--------|------|-----|-----|
| 1678_1849 | | | | | | | | | | |
| 1850_1899 | | | | | | | | | | |
| 1900_1949 | | | | | | | | | | |
| 1950_1999 | | | | | | | | | | |
| 2000_2009 | | | | | | | | | | |
| 2010_2023 | | | | | | | | | | |

---

## Dependencies

- **Required:** `openai`, `pandas`, `pyarrow`
- **Optional:** `datasketch` (for MinHash near-duplicate detection)
