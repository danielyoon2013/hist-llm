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
D     Quantitative      mc4, open, cot                3
E     Completion        mc4                           1
F     Instruct          mc4_passage                   1
                                              Total: 10 slots
```

At `--target 1,000,000`:

```
per_slot = 1,000,000 / 10 = 100,000

A: 100,000 x 2 = 200,000
B: 100,000 x 2 = 200,000
C: 100,000 x 1 = 100,000
D: 100,000 x 3 = 300,000
E: 100,000 x 1 = 100,000
F: 100,000 x 1 = 100,000
                ─────────
         Total: 1,000,000
```

### From Target to Docs Needed

Each document is chunked (~2 chunks per doc at 6000 chars). Each chunk produces 2 items per API call (`ITEMS_PER_CALL = 2`). So:

```
items_per_doc = ITEMS_PER_CALL x CHUNKS_PER_DOC = 2 x 2 = 4

docs_needed (per generator) = ceil(per_slot / items_per_doc)
                             = ceil(100,000 / 4) = 25,000
```

Each generator independently samples its own docs from `synthetic/input/`. All of this is computed by `compute_plan()` — you only specify `--target`.

### Per-Period Output Targets

| Output | Size | Description |
|--------|-----:|-------------|
| Mid-train | ~950,000 | All formats from 95% of docs |
| SFT | ~9,500 | ~1% proportional subsample from train |
| Test | ~50,000 | MC-only from 5% held-out docs |

---

## The 6 Generators (10 Format Slots)

All generators are corpus-based — they read documents from `synthetic/input/` and produce training conversations grounded in historical text.

| ID | Name | Formats | Benchmark Alignment | Items/Call |
|----|------|---------|---------------------|-----------|
| A | Factual QA | `mc4`, `open` | ARC | 2 |
| B | Chain-of-Thought | `mc4`, `cot` | ARC | 2 |
| C | Reading Comprehension | `mc4_passage` | RACE | 2 |
| D | Quantitative | `mc4`, `open`, `cot` | — | 2 |
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

Each line is a metadata-wrapped conversation (for document-level splitting):
```jsonl
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"doc_name":"collection/doc_00001","chunk_idx":0,"generator":"gen_a_factual","format":"mc4"}
```

### Step 3: Assemble (Document-Level Train/Test Split)

Performs a **document-level** split: 95% of unique documents → train (all formats), 5% → test (MC-only). No content leakage between train and test.

```bash
python -m src.post_training.assemble --period 1900_1949 --source raw
python -m src.post_training.assemble --period 1900_1949 --dry-run  # preview only
```

**Output:**
- `final/train/hist_synthetic_midtrain.jsonl` — all train examples, all formats (bare message lists)
- `final/train/hist_synthetic_sft.jsonl` — 10K proportional subsample
- `final/test/hist_synthetic_test.jsonl` — MC-only from held-out docs (with `letters` field for eval)

---

## API Usage and Cost

Generation uses `gpt-4o-mini` via the OpenAI Batch API (50% discount).

At 1M target with 10 format slots and ~25,000 docs per generator:
- ~6 generators x ~25,000 docs x ~2 chunks = ~300,000 API calls
- At ~$0.00075/call (Batch API average) = **~$225/period**
- x 6 periods = **~$1,350 total**

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
│   ├── gen_d_quantitative.py          # D: Quantitative / Math (mc4, open, cot)
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
│   │   ├── gen_d_quantitative_mc4.jsonl
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
│   │   ├── hist_synthetic_midtrain.jsonl  # All formats from train docs (~950K)
│   │   └── hist_synthetic_sft.jsonl       # ~1% proportional subsample (~9.5K)
│   └── test/
│       └── hist_synthetic_test.jsonl      # MC-only from held-out docs (~50K)
│
└── quality/                           # Quality pipeline outputs (deferred for v1)
    ├── validated/
    └── deduped/
```

---

## nanochat Integration

**Training files** (`midtrain`, `sft`) use nanochat's **CustomJSON** format (bare message lists):

```jsonl
[{"role":"user","content":"What caused...?"},{"role":"assistant","content":"The main cause..."}]
[{"role":"user","content":"Multiple Choice question: ...\n- choice=A\n- choice=B\n..."},{"role":"assistant","content":"B"}]
```

**Test file** uses nanochat's **categorical eval** format (matches LABEval):

```jsonl
{"messages":[{"role":"user","content":"Multiple Choice question: ..."},{"role":"assistant","content":"B"}],"letters":["A","B","C","D"]}
```

Load as a custom eval task (same pattern as `tasks/lab_eval.py`).

---

## Evaluation and Train/Test Strategy

### Document-Level Train/Test Split

`assemble.py` performs a **document-level split** — 95% of unique documents go to train (all formats), 5% go to test (MC-only). This ensures zero content leakage between train and test.

| Output | Size | Format | Purpose |
|--------|-----:|--------|---------|
| `hist_synthetic_midtrain.jsonl` | ~950K | All formats (mc4, open, cot, passage) | Mid-training |
| `hist_synthetic_sft.jsonl` | ~9,500 | Proportional subsample (~1% of train) | SFT |
| `hist_synthetic_test.jsonl` | ~50K | MC-only (`mc4`, `mc4_passage`) | Categorical eval |

**Why document-level?** If the same chunk produces both `open` and `mc4` formats, putting `open` in train and `mc4` in test creates content leakage — the model has seen the answer during training. By splitting at the document level, test questions come from documents the model has never seen in any format.

**Why MC-only test?** nanochat's categorical eval is fast (batched logit comparison, no sampling). This enables evaluation every N training steps without significant overhead. Non-MC formats from test documents are discarded (~5% waste).

**Format diversity in training:** Train documents contribute ALL formats (mc4, open, cot, mc4_passage). This ensures the model learns diverse response patterns during mid-training and SFT.

### 3-Source Evaluation

We evaluate with three sources, each testing a different axis:

#### Source 1: Internal MC Test Split (did the model learn historical content?)

Our `hist_synthetic_test.jsonl` — MC questions from held-out documents. Loaded as a categorical task in nanochat (same format as LABEval).

#### Source 2: External Benchmarks (does the model retain general capabilities?)

Time-invariant benchmarks already in nanochat. These monitor catastrophic forgetting — scores should stay stable during training, not necessarily improve.

| Benchmark | Format | Contamination | What It Tests | In nanochat? |
|-----------|--------|--------------|---------------|-------------|
| ARC-Challenge | MC-4 | ~2% | Science reasoning | Yes |
| HellaSwag | MC-4 | Low (est.) | Commonsense completion | Yes |
| RACE-Middle | MC-4 + Passage | Low (est.) | Reading comprehension | Yes |
| RACE-High | MC-4 + Passage | Low (est.) | Reading comprehension | Yes |
| Winogrande | MC-2 | Low (est.) | Coreference resolution | Yes |

**Why not MMLU?** 34.6% LAB contamination (1950-1999 period). Too much post-period knowledge to serve as a time-invariant benchmark.

**Why not GSM8K?** It's generative eval (sequential sampling) — too slow for frequent training-interval evaluation. Our Gen D MC4 format covers math reasoning in the internal test split.

#### Source 3: LAB Eval (temporal isolation — the core thesis)

5,000 MC questions per period about events AFTER the period's end year. A perfectly isolated model scores 25% (random chance).

| Metric | Target | Meaning |
|--------|--------|---------|
| LAB accuracy | ~25% | Random chance = no future knowledge |
| LAP score | < 0.05 | `(LAB_acc - 0.25) / 0.75` — 0 is perfect |

### Evaluation Schedule

All three sources use categorical (MC) format — fast, batched, no sampling.

| Training Stage | Frequency | What to Evaluate |
|----------------|-----------|------------------|
| Base training | Every 2000 steps | CORE metric (base_eval.py) |
| Mid-training | Every 200 steps | Internal MC + HellaSwag + Winogrande |
| Mid-training | At boundaries | All 3 sources (full 5-benchmark + LAB) |
| SFT | Every 200 steps | Internal MC + HellaSwag + Winogrande |
| SFT | At boundaries | All 3 sources |
| 1900_1949 | | | | | | | | | | |
| 1950_1999 | | | | | | | | | | |
---

## Dependencies

- **Required:** `openai`, `pandas`, `pyarrow`
- **Optional:** `datasketch` (for MinHash near-duplicate detection)
